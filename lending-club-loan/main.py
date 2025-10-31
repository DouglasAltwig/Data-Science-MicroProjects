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


def set_champion_alias(model_name, alias):
    """Sets an alias for the latest version of the registered model."""
    print(f"\n--- Setting '{alias}' alias for the best model ---")
    client = MlflowClient()
    try:
        latest_version_info = client.search_model_versions(f"name='{model_name}'")[0]
        latest_version = latest_version_info.version
        client.set_registered_model_alias(model_name, alias, latest_version)
        print(f"Alias '{alias}' set for version {latest_version} of model '{model_name}'.")
    except IndexError:
        print(f"Warning: No model version found in stage 'None' for '{model_name}'. Alias not set.")


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

    # ==============================
    # --- Hyperparameter Tuning  ---
    # ==============================
    print(f"\n--- Starting Hyperparameter Tuning ---")
    
    # param_grid = {
    #     'module__layer_sizes': [[64,128, 64], [256, 128]],
    #     'module__dropout_rate': [0.2, 0.4],
    #     'lr': [1e-4, 1e-3],
    #     'optimizer__weight_decay': [1e-5, 1e-4],
    #     'batch_size': [256, 512]
    # }

    param_grid = {
        'module__layer_sizes': [[256, 128, 64, 32, 16]],
        'module__dropout_rate': [0.4],
        'lr': [1e-3],
        'optimizer__weight_decay': [1e-5],
        'batch_size': [2048]
    }

    params_list = list(ParameterGrid(param_grid))
    best_f1 = -1
    best_params = None
    
    for i, params in enumerate(params_list):
        print(f"\nTrail {i+1}/{len(params_list)}: {params}")

        net = NeuralNetBinaryClassifier(
            module=LoanPredictorDNN,
            module__input_size=n_features,
            criterion=nn.BCEWithLogitsLoss,
            criterion__pos_weight=pos_weight,
            optimizer=torch.optim.AdamW,
            device=config.DEVICE,
            max_epochs=config.HPO_EPOCHS,
            train_split=ValidSplit(cv=0.1, stratified=False), # Internal split for skorch fit
            callbacks=[
                EarlyStopping(patience=config.HPO_PATIENCE),
                LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=3)
            ],
            verbose=0
        ).set_params(**params)
    
        # Fit on training data
        X_train_np = X_train.astype(np.float32)
        y_train_np = y_train.to_numpy().astype(np.float32)
        net.fit(X_train_np, y_train_np)

        # Evaluate on the dedicated, held-out validation set
        X_val_np = X_val.astype(np.float32)
        y_val_probs = net.predict_proba(X_val_np)[:, 1]
        
        optimal_threshold = find_optimal_threshold(y_val, y_val_probs)
        y_val_pred = (y_val_probs >= optimal_threshold).astype(int)
        val_f1 = f1_score(y_val, y_val_pred, pos_label=0)
        print(f"Validation F1-score (Bad Loan) at optimal threshold {optimal_threshold}: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params
            print(f"New best model found with F1-score: {best_f1:.4f}")
    
    print(f"\nBest hyperparameters found: {best_params} with F1-score: {best_f1:.4f}")

    # ===========================================
    # --- Champion Model Training and Logging ---
    # ===========================================
    print("\n--- Training and Logging Champion Model ---")
    print(f"Champion model will use {n_features} features.")

    # Combine original training and validation sets for final training
    X_train_full = np.vstack((X_train, X_val))
    y_train_full = pd.concat([y_train, y_val])
    y_train_full_np = y_train_full.to_numpy().astype(np.float32)
    
    # Create a predefined split for skorch to use the original validation set for EarlyStopping
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
        train_split=predefined_split
    ).set_params(**best_params)

    print("Fitting the champion model on the combined training and validation data...")
    champion_classifier.fit(X_train_full.astype(np.float32), y_train_full_np)

    # Find optimal threshold using the original validation data
    y_val_probs_champion = champion_classifier.predict_proba(X_val.astype(np.float32))[:, 1]
    optimal_threshold_champion = find_optimal_threshold(y_val, y_val_probs_champion)
    print(f"Found optimal threshold for champion model: {optimal_threshold_champion}")

    # Log everything to MLflow
    with mlflow.start_run(run_name="Champion_Model_Pipeline") as run:
        mlflow.log_params(best_params)
        mlflow.log_params({
            "optimal_threshold": optimal_threshold_champion,
            "final_feature_count": n_features
        })
        
        # Build a complete scikit-learn pipeline for deployment
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', ThresholdClassifier(model=champion_classifier, threshold=optimal_threshold_champion))
        ])
        
        # Create model signature for MLflow
        input_schema_list = []
        for col_name, dtype in X_test_raw.dtypes.items():
            if np.issubdtype(dtype, np.integer):
                # Use "long" for standard integers (e.g., int64)
                mlflow_type = "long"
            elif np.issubdtype(dtype, np.floating):
                # Use "double" for standard floats (e.g., float64)
                mlflow_type = "double"
            elif np.issubdtype(dtype, np.object_):
                # Pandas 'object' dtypes are typically strings
                mlflow_type = "string"
            else:
                # A fallback for other types like boolean, etc.
                mlflow_type = str(dtype)

            input_schema_list.append(ColSpec(mlflow_type, col_name))
        
        input_schema = Schema(input_schema_list)
        output_schema = Schema([TensorSpec(np.dtype(np.int64), (-1,), "prediction")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Log the pipeline model
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_test_raw.iloc[:5],
            registered_model_name=config.REGISTERED_MODEL_NAME
        )
        
        # Evaluate on test set and log artifacts
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
        
        print(f"Champion model logged to run ID: {run.info.run_id}")
    
    # --- Set Champion Alias in Model Registry ---
    set_champion_alias(config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS)

    # Example: Load and predict with the newly registered champion model
    print("\n--- Loading champion model pipeline from MLflow for prediction ---")
    model_uri = f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
    loaded_pipeline = mlflow.pyfunc.load_model(model_uri)
    print("Champion model pipeline loaded successfully.")
    
    sample = X_test_raw.iloc[:1]
    prediction = loaded_pipeline.predict(sample)[0]
    print(f"Example prediction on a raw sample: {'Bad Loan' if prediction == 1 else 'Good Loan'}")

if __name__ == "__main__":
    main()
