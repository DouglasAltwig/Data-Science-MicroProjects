import os
import torch
import torch.nn as nn

# --- Data and Artifacts ---
DATA_FILEPATH = "./data/accepted_2007_to_2018Q4.csv"
PREPROCESSED_PARQUET_FILEPATH = "./data/processed/engineered_features.parquet"
ARTIFACT_PATH = "artifacts"

# --- Model & Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION_FN = nn.BCEWithLogitsLoss # Loss function class

# Champion model & training parameters
REGISTERED_MODEL_NAME = "lending_club_loan_classification_model"
CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"
CHAMPION_EPOCHS = 200
CHAMPION_PATIENCE = 15

# HPO model training parameters
HPO_EPOCHS = 50
HPO_PATIENCE = 5

# --- Data Split Configuration ---
TRAIN_END_DATE = '2016-01-01'
VAL_END_DATE = '2017-01-01'