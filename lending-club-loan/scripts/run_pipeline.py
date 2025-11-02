import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import torch
import mlflow
import numpy as np

from src import config
from src.data_processing.loader import load_and_clean_data
from src.data_processing.preprocess import feature_engineer, preprocess_dataset
from src.model.trainer import run_hpo, train_final_model
from src.utils import set_model_alias

def main():
    """Main function to orchestrate the entire ML pipeline."""
    print(f"Using device: {config.DEVICE}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # --- Data Loading and Caching ---
    if os.path.exists(config.PREPROCESSED_PARQUET_FILEPATH):
        print(f"Loading cached engineered data from {config.PREPROCESSED_PARQUET_FILEPATH}...")
        df_engineered = pd.read_parquet(config.PREPROCESSED_PARQUET_FILEPATH)
    else:
        print("Cached data not found. Processing from raw CSV...")
        df = load_and_clean_data(config.DATA_FILEPATH)
        df_engineered = feature_engineer(df)
        os.makedirs(os.path.dirname(config.PREPROCESSED_PARQUET_FILEPATH), exist_ok=True)
        print(f"Saving engineered data to {config.PREPROCESSED_PARQUET_FILEPATH}...")
        df_engineered.to_parquet(config.PREPROCESSED_PARQUET_FILEPATH, index=False)

    # --- Data Splitting and Preprocessing ---
    X_train, X_val, X_test, y_train, y_val, y_test, X_test_raw, preprocessor = preprocess_dataset(
        df_engineered, config.TRAIN_END_DATE, config.VAL_END_DATE
    )
    
    # Convert data to float32 for PyTorch/skorch
    X_train_torch = X_train.astype(np.float32)
    X_val_torch = X_val.astype(np.float32)
    X_test_torch = X_test.astype(np.float32)

    with mlflow.start_run(run_name="Loan_Default_Prediction_Pipeline") as parent_run:
        print(f"\n--- Started Parent Run: {parent_run.info.run_id} ---")
        
        # --- Hyperparameter Tuning ---
        n_features = X_train.shape[1]
        class_counts = y_train.value_counts()
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=config.DEVICE)
        
        best_params = run_hpo(X_train_torch, y_train, X_val_torch, y_val, n_features, pos_weight)

        # --- Champion Model Training ---
        champion_run_id = train_final_model(
            best_params, X_train_torch, y_train, X_val_torch, y_val, 
            X_test_torch, y_test, X_test_raw, preprocessor
        )

    if champion_run_id:
        # --- Set Candidate Alias ---
        set_model_alias(config.REGISTERED_MODEL_NAME, config.CANDIDATE_ALIAS, champion_run_id)

        print("\n" + "="*50)
        print("PIPELINE FINISHED")
        print(f"A new model candidate has been registered from run: {champion_run_id}")
        print("To promote this model, run: python scripts/promote_model.py")
        print("="*50 + "\n")
    else:
        print("Pipeline finished without training a new model.")


if __name__ == "__main__":
    main()