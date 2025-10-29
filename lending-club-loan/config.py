import os
import torch
import torch.nn as nn

# --- Data and Artifacts ---
DATA_FILEPATH = "./data/accepted_2007_to_2018Q4.csv"
PREPROCESSED_PARQUET_FILEPATH = "./data/processed/engineered_features.parquet"
ARTIFACT_PATH = "artifacts"
PREPROCESSOR_FILEPATH = os.path.join(ARTIFACT_PATH, "preprocessor.joblib")

# --- Model & Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION_FN = nn.BCEWithLogitsLoss # Loss function class
NUM_DATALOADER_WORKERS = 0 #os.cpu_count() - 1 if os.cpu_count() > 1 else 0

# Champion model training parameters (can be larger than HPO)
CHAMPION_EPOCHS = 200
CHAMPION_PATIENCE = 15
CHAMPION_BATCH_SIZE = 256

# HPO model training parameters
HPO_EPOCHS = 50
HPO_PATIENCE = 5
HPO_BATCH_SIZE = 256

# --- Data Split Configuration ---
TRAIN_END_DATE = '2016-01-01'
VAL_END_DATE = '2017-01-01'