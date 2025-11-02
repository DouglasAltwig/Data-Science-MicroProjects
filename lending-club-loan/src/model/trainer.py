import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
import mlflow

from src import config
from src.model.definition import LoanPredictorDNN
from src.model.pipeline import ThresholdClassifier
from src.evaluation.metrics import find_optimal_threshold, evaluate_champion_model
from src.utils import create_model_signature

def run_hpo(X_train, y_train, X_val, y_val, n_features, pos_weight):
    """Runs hyperparameter optimization and returns the best parameters."""
    print("\n--- Starting Hyperparameter Tuning (Child Runs) ---")
    
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
                criterion=config.CRITERION_FN,
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
    
            net.fit(X_train, y_train.to_numpy().astype(np.float32))
            y_val_probs = net.predict_proba(X_val)[:, 1]
            
            y_val_pred = (y_val_probs >= 0.5).astype(int)
            val_f1 = f1_score(y_val, y_val_pred, pos_label=1)
            
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

    return best_params

def train_final_model(best_params, X_train, y_train, X_val, y_val, X_test, y_test, X_test_raw, preprocessor):
    """Trains, evaluates, and logs the final champion model."""
    if not best_params:
        print("HPO did not find any successful parameters. Exiting.")
        return None

    print("\n--- Training and Logging Champion Model (Child Run) ---")
    with mlflow.start_run(run_name="Champion_Model_Training", nested=True) as champion_run:
        champion_run_id = champion_run.info.run_id
        n_features = X_train.shape[1]
        class_counts = y_train.value_counts()
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=config.DEVICE)

        print(f"Champion model will use {n_features} features. Run ID: {champion_run_id}")
        mlflow.log_param("champion_model_run_id", champion_run_id)

        X_train_full = np.vstack((X_train, X_val))
        y_train_full_torch = pd.concat([y_train, y_val]).to_numpy().astype(np.float32)
        
        train_indices = np.arange(len(X_train))
        val_indices = np.arange(len(X_train), len(X_train_full))
        predefined_split = ValidSplit(cv=[(train_indices, val_indices)])

        champion_classifier = NeuralNetBinaryClassifier(
            module=LoanPredictorDNN,
            module__input_size=n_features,
            criterion=config.CRITERION_FN,
            criterion__pos_weight=pos_weight,
            optimizer=torch.optim.AdamW,
            device=config.DEVICE,
            max_epochs=config.CHAMPION_EPOCHS,
            callbacks=[EarlyStopping(patience=config.CHAMPION_PATIENCE)],
            train_split=predefined_split,
            verbose=0
        ).set_params(**best_params)

        print("Fitting the champion model on the combined training and validation data...")
        champion_classifier.fit(X_train_full, y_train_full_torch)

        y_val_probs_champion = champion_classifier.predict_proba(X_val)[:, 1]
        optimal_threshold = find_optimal_threshold(y_val, y_val_probs_champion)

        mlflow.log_params(best_params)
        mlflow.log_params({"optimal_threshold": optimal_threshold, "final_feature_count": n_features})
        
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', ThresholdClassifier(model=champion_classifier, threshold=optimal_threshold))
        ])
        
        signature = create_model_signature(X_test_raw)
        
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_test_raw.iloc[:5],
            registered_model_name=config.REGISTERED_MODEL_NAME
        )
        
        os.makedirs(config.ARTIFACT_PATH, exist_ok=True)
        eval_metrics = evaluate_champion_model(
            champion_classifier.module_, X_test, y_test, 
            config.DEVICE, optimal_threshold, config.ARTIFACT_PATH
        )
        mlflow.log_metrics({
            "test_roc_auc": eval_metrics["test_roc_auc"],
            "test_pr_auc_bad_loan": eval_metrics["test_pr_auc"]
        })
        mlflow.log_artifacts(config.ARTIFACT_PATH)
        
        print(f"Champion model logged to run ID: {champion_run_id}")
        return champion_run_id