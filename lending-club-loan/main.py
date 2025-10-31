import os
import numpy as np
import pandas as pd

# --- scikit-learn imports ---
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# --- PyTorch imports ---
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- skorch imports ---
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

# --- MLflow imports ---
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec

# --- Local Imports ---
import config
from data_loader import load_and_clean_data
from preprocess import feature_engineer, preprocess_dataset
from evaluate import evaluate_champion_model, find_optimal_threshold
from pipeline import ThresholdClassifier
from models import LoanPredictorDNN


def set_model_alias(model_name, alias, run_id):
    """Sets an alias for the model version created by a specific run."""
    print(f"\n--- Setting '{alias}' alias for model from run {run_id} ---")
    client = MlflowClient()
    try:
        # Find the model version associated with the champion model's run_id
        results = client.search_model_versions(f"name='{model_name}' AND run_id='{run_id}'")
        if not results:
            print(f"Warning: No model version found for '{model_name}' from run '{run_id}'. Alias not set.")
            return

        latest_version = results[0].version
        client.set_registered_model_alias(model_name, alias, latest_version)
        print(f"Alias '{alias}' set for version {latest_version} of model '{model_name}'.")
    except Exception as e:
        print(f"Error setting alias for model '{model_name}': {e}")


def main():
    """Main function to run the entire ML pipeline."""

    print(f"Using device: {config.DEVICE}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # ================================
    # --- Data Loading and Caching ---
    # ================================
    if os.path.exists(config.PREPROCESSED_PARQUET_FILEPATH):
        print(f"Loading cached engineered data from {config.PREPROCESSED_PARQUET_FILEPATH}...")
        df_engineered = pd.read_parquet(config.PREPROCESSED_PARQUET_FILEPATH)
    else:
        print("Cached engineered data not found. Processing from raw CSV...")
        df = load_and_clean_data(config.DATA_FILEPATH)
        df_engineered = feature_engineer(df)
        print(f"Saving engineered data to {config.PREPROCESSED_PARQUET_FILEPATH} for caching...")
        df_engineered.to_parquet(config.PREPROCESSED_PARQUET_FILEPATH, index=False)

    # ================================
    # Data Splitting and Preprocessing
    # ================================
    X_train, X_val, X_test, y_train, y_val, y_test, X_test_raw, preprocessor = preprocess_dataset(df_engineered, config.TRAIN_END_DATE, config.VAL_END_DATE)
    
    # --- Shared variables for model training ---
    n_features = X_train.shape[1]
    class_counts = y_train.value_counts()
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=config.DEVICE)

    with mlflow.start_run(run_name="Loan_Default_Prediction_Pipeline") as parent_run:
        print(f"\n--- Started Parent Run: {parent_run.info.run_id} ---")
        
        # ==============================
        # --- Hyperparameter Tuning  ---
        # ==============================
        print(f"\n--- Starting Hyperparameter Tuning (Child Runs) ---")
        
        param_grid = {
            'module__layer_sizes': [[256, 128, 64, 32, 16], [128, 64, 32]],
            'module__dropout_rate': [0.3, 0.4],
            'lr': [1e-3, 5e-4],
            'optimizer__weight_decay': [1e-5],
            'batch_size': [2048]
        }

        params_list = list(ParameterGrid(param_grid))
        best_f1 = -1
        best_params = None
        
        for i, params in enumerate(params_list):
            with mlflow.start_run(run_name=f"hpo_trial_{i+1}", nested=True) as child_run:
                print(f"\nTrial {i+1}/{len(params_list)}: {params} (Run ID: {child_run.info.run_id})")
                mlflow.log_params(params)

                net = NeuralNetBinaryClassifier(
                    module=LoanPredictorDNN,
                    module__input_size=n_features,
                    criterion=nn.BCEWithLogitsLoss,
                    criterion__pos_weight=pos_weight,
                    optimizer=torch.optim.AdamW,
                    device=config.DEVICE,
                    max_epochs=config.HPO_EPOCHS,
                    train_split=ValidSplit(cv=0.1, stratified=False),
                    callbacks=[
                        EarlyStopping(patience=config.HPO_PATIENCE),
                        LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=3)
                    ],
                    verbose=0
                ).set_params(**params)
        
                X_train_np = X_train.astype(np.float32)
                y_train_np = y_train.to_numpy().astype(np.float32)
                net.fit(X_train_np, y_train_np)

                X_val_np = X_val.astype(np.float32)
                y_val_probs = net.predict_proba(X_val_np)[:, 1]
                
                y_val_pred = (y_val_probs >= 0.5).astype(int)
                val_f1 = f1_score(y_val, y_val_pred, pos_label=1) # F1 for "Bad Loan" (class 1)
                
                mlflow.log_metric("validation_f1_score", val_f1)
                print(f"Validation F1-score (Bad Loan): {val_f1:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_params = params
                    print(f"New best model found with F1-score: {best_f1:.4f}")
        
        print("\n--- HPO Finished ---")
        print(f"Best hyperparameters found: {best_params} with F1-score: {best_f1:.4f}")
        if best_params:
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_validation_f1_score", best_f1)
        
        if not best_params:
            print("HPO did not find any successful parameters. Exiting.")
            return

        # ===========================================
        # --- Champion Model Training and Logging ---
        # ===========================================
        print("\n--- Training and Logging Champion Model (Child Run) ---")

        with mlflow.start_run(run_name="Champion_Model_Training", nested=True) as champion_run:
            champion_run_id = champion_run.info.run_id
            print(f"Champion model will use {n_features} features. Run ID: {champion_run_id}")
            mlflow.log_param("champion_model_run_id", champion_run_id)

            X_train_full = np.vstack((X_train, X_val))
            y_train_full = pd.concat([y_train, y_val])
            y_train_full_np = y_train_full.to_numpy().astype(np.float32)
            
            n_train = len(X_train)
            train_indices = np.arange(n_train)
            val_indices = np.arange(n_train, len(X_train_full))
            predefined_split = ValidSplit(cv=[(train_indices, val_indices)])

            champion_classifier = NeuralNetBinaryClassifier(
                module=LoanPredictorDNN,
                module__input_size=n_features,
                criterion=nn.BCEWithLogitsLoss,
                criterion__pos_weight=pos_weight,
                optimizer=torch.optim.AdamW,
                device=config.DEVICE,
                max_epochs=config.CHAMPION_EPOCHS,
                callbacks=[EarlyStopping(patience=config.CHAMPION_PATIENCE)],
                train_split=predefined_split,
                verbose=0
            ).set_params(**best_params)

            print("Fitting the champion model on the combined training and validation data...")
            champion_classifier.fit(X_train_full.astype(np.float32), y_train_full_np)

            y_val_probs_champion = champion_classifier.predict_proba(X_val.astype(np.float32))[:, 1]
            optimal_threshold_champion = find_optimal_threshold(y_val, y_val_probs_champion)
            print(f"Found optimal threshold for champion model: {optimal_threshold_champion}")

            mlflow.log_params(best_params)
            mlflow.log_params({
                "optimal_threshold": optimal_threshold_champion,
                "final_feature_count": n_features
            })
            
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', ThresholdClassifier(model=champion_classifier, threshold=optimal_threshold_champion))
            ])
            full_pipeline._is_fitted = True
            
            input_schema_list = []
            for col, dtype in X_test_raw.dtypes.items():
                if np.issubdtype(dtype, np.floating):
                    mlflow_type = "double"  # 'float64' maps to 'double'
                elif np.issubdtype(dtype, np.integer):
                    mlflow_type = "long"    # 'int64' maps to 'long'
                else:
                    mlflow_type = "string"  # 'object' and others map to 'string'
                input_schema_list.append(ColSpec(type=mlflow_type, name=col))
            
            input_schema = Schema(input_schema_list)
            output_schema = Schema([TensorSpec(np.dtype(np.int64), (-1,), "prediction")])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            
            mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                name="model", # `artifact_path` is deprecated
                signature=signature,
                input_example=X_test_raw.iloc[:5],
                registered_model_name=config.REGISTERED_MODEL_NAME
            )
            
            os.makedirs(config.ARTIFACT_PATH, exist_ok=True)
            eval_metrics = evaluate_champion_model(
                champion_classifier.module_, X_test, y_test, 
                config.DEVICE, optimal_threshold_champion, config.ARTIFACT_PATH
            )
            mlflow.log_metrics({
                "test_roc_auc": eval_metrics["test_roc_auc"],
                "test_pr_auc_bad_loan": eval_metrics["test_pr_auc"]
            })
            mlflow.log_artifacts(config.ARTIFACT_PATH)
            
            print(f"Champion model logged to run ID: {champion_run_id}")
    
    set_model_alias(config.REGISTERED_MODEL_NAME, config.CANDIDATE_ALIAS, champion_run_id)

    print("\n" + "="*50)
    print("PIPELINE FINISHED")
    print(f"A new model candidate has been registered from run: {champion_run_id}")
    print(f"To promote this model to production, run the `promote_model.py` script")
    print("or manually change the 'champion' alias in the MLflow UI.")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()