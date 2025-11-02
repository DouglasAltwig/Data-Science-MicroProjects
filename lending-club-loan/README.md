# Lending Club Loan Default Prediction

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-blue)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)

An end-to-end MLOps project for predicting loan defaults using Lending Club data. This repository demonstrates a complete workflow from data processing and model training to deployment with a live-reloading inference API.

![Lending Club Loan Default Prediction Project Banner](../assets/lendingclubloan.jpg)

## Table of Contents
- [Lending Club Loan Default Prediction](#lending-club-loan-default-prediction)
  - [Table of Contents](#table-of-contents)
  - [✨ Features](#-features)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation \& Setup](#installation--setup)
  - [⚙️ Usage](#️-usage)
    - [1. Run the Training Pipeline](#1-run-the-training-pipeline)
    - [2. Promote the Model to Production](#2-promote-the-model-to-production)
    - [3. Test the Inference API](#3-test-the-inference-api)
    - [4. Stop the Environment](#4-stop-the-environment)
  - [🏗️ Project Architecture](#️-project-architecture)
    - [Directory Structure](#directory-structure)
    - [Core Components \& Logic](#core-components--logic)
  - [🛠️ Technology Stack](#️-technology-stack)

## ✨ Features

*   **End-to-End MLOps Pipeline**: Covers data loading, cleaning, feature engineering, and training.
*   **Automated Hyperparameter Tuning**: Uses Optuna for hyperparameter optimization, with results tracked as nested MLflow runs.
*   **Experiment Tracking with MLflow**: All experiments, parameters, metrics, and models are logged and versioned.
*   **Model Registry for Governance**: Manages model lifecycle with `candidate` and `champion` aliases.
*   **Live Inference Service with FastAPI**: A production-ready API that serves the champion model.
*   **Automatic Model Polling**: The FastAPI service automatically detects and deploys new champion models from the MLflow registry without needing a restart.
*   **Reproducible Environment**: Fully containerized with Docker Compose for consistent setup across machines.

## 🚀 Getting Started

Follow these steps to get the project environment up and running on your local machine.

### Prerequisites

*   [Git](https://git-scm.com/)
*   [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose
*   NVIDIA GPU with drivers and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed.

### Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/DouglasAltwig/Data-Science-MicroProjects.git
    ```

2.  **Navigate to the Project Directory**
    ```bash
    cd Data-Science-MicroProjects/lending-club-loan
    ```
    *Note: All subsequent commands must be run from this directory.*

3.  **Start the Services with Docker Compose**
    This command will build and start all services (MLflow, MinIO, Postgres, Jupyter, FastAPI) in the background.
    ```bash
    docker-compose up -d
    ```
    > **Note on First-Time Startup:** The initial launch can take **5-10 minutes** as Docker downloads the base images and the container installs Python dependencies. You can monitor the progress with `docker-compose logs -f`.

4.  **Access the JupyterLab Environment**
    Once the containers are running, open your browser and navigate to **[http://localhost:8888](http://localhost:8888)**. You will find the project files in the `extra/` directory inside JupyterLab.

## ⚙️ Usage

After setting up the environment, you can run the machine learning pipeline and interact with the deployed model.

### 1. Run the Training Pipeline

Execute the main pipeline script. This will process the data, run hyperparameter tuning, train the final model, and register it in MLflow with the `candidate` alias.

```bash
python scripts/run_pipeline.py
```
You can view the experiment runs and the newly registered model in the MLflow UI at **[http://localhost:5000](http://localhost:5000)**.

### 2. Promote the Model to Production

After reviewing the `candidate` model's performance in MLflow, promote it to the `champion` stage. This makes it the production-ready model that the API will serve.

```bash
python scripts/promote_model.py
```

### 3. Test the Inference API

The FastAPI service (running on port 8000) will automatically detect the new `champion` model within 60 seconds. You can send a `POST` request to the `/predict` endpoint to get a prediction.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "loan_amnt": 10000,
    "term": " 36 months",
    "int_rate": 11.44,
    "installment": 329.47,
    "grade": "B",
    "sub_grade": "B4",
    "emp_title": "teacher",
    "emp_length": "10+ years",
    "home_ownership": "RENT",
    "annual_inc": 117000,
    "verification_status": "Not Verified",
    "issue_d": "Jan-2015",
    "purpose": "vacation",
    "title": "Vacation",
    "dti": 6.08,
    "earliest_cr_line": "Feb-1990",
    "open_acc": 16,
    "pub_rec": 0,
    "revol_bal": 21320,
    "revol_util": 53.9,
    "total_acc": 35,
    "initial_list_status": "w",
    "collections_12_mths_ex_med": 0,
    "policy_code": 1,
    "application_type": "Individual",
    "acc_now_delinq": 0,
    "tot_coll_amt": 0,
    "tot_cur_bal": 34360,
    "total_rev_hi_lim": 39500,
    "acc_open_past_24mths": 1,
    "avg_cur_bal": 2148,
    "bc_open_to_buy": 1506,
    "bc_util": 93.7,
    "chargeoff_within_12_mths": 0,
    "delinq_amnt": 0,
    "mo_sin_old_il_acct": 141,
    "mo_sin_old_rev_tl_op": 299,
    "mo_sin_rcnt_rev_tl_op": 1,
    "mo_sin_rcnt_tl": 1,
    "mort_acc": 0,
    "mths_since_recent_bc": 5,
    "mths_since_recent_inq": 4,
    "num_accts_ever_120_pd": 0,
    "num_actv_bc_tl": 5,
    "num_actv_rev_tl": 8,
    "num_bc_sats": 6,
    "num_bc_tl": 11,
    "num_il_tl": 13,
    "num_op_rev_tl": 11,
    "num_rev_accts": 22,
    "num_rev_tl_bal_gt_0": 8,
    "num_sats": 16,
    "num_tl_120dpd_2m": 0,
    "num_tl_30dpd": 0,
    "num_tl_90g_dpd_24m": 0,
    "num_tl_op_past_12m": 1,
    "pct_tl_nvr_dlq": 100,
    "percent_bc_gt_75": 100,
    "pub_rec_bankruptcies": 0,
    "tax_liens": 0,
    "total_bal_ex_mort": 34360,
    "total_bc_limit": 24000,
    "total_il_high_credit_limit": 18354
  }'
```

The service also has a health check endpoint at **[http://localhost:8000/health](http://localhost:8000/health)**.

### 4. Stop the Environment

When you are finished, shut down all the containers to free up system resources.

```bash
docker-compose down
```

## 🏗️ Project Architecture

### Directory Structure

```
.
├── deployment/fastapi_service/  # FastAPI application for serving
│   ├── main.py                # API endpoints (/predict, /health) & model polling logic
│   └── Dockerfile               # Container setup for the API
├── scripts/
│   ├── run_pipeline.py          # Main entrypoint to run the E2E training pipeline
│   └── promote_model.py         # Utility to promote a 'candidate' model to 'champion'
├── src/
│   ├── data_processing/         # Data loading, cleaning, and feature engineering
│   ├── model/                   # Model definition, training logic, and custom pipeline components
│   └── utils.py                 # Helper functions (e.g., for MLflow)
├── docker-compose.yml           # Orchestrates all services (MLflow, API, etc.)
└── README.md
```

### Core Components & Logic

<details>
<summary><strong>Training Pipeline (<code>scripts/run_pipeline.py</code>)</strong></summary>

This script orchestrates the entire ML workflow:
1.  **Load Data**: Loads the raw CSV and performs initial cleaning via `load_and_clean_data`.
2.  **Feature Engineering**: Creates new features from existing ones using `feature_engineer`.
3.  **Data Splitting & Preprocessing**: Splits data chronologically and applies preprocessing transformations via `preprocess_dataset`.
4.  **Hyperparameter Optimization**: Runs `run_hpo` to find the best model parameters. Each trial is logged as a nested MLflow run.
5.  **Final Model Training**: Trains the final model on the full training set using the best hyperparameters found in the HPO step.
6.  **MLflow Logging**: The entire scikit-learn pipeline (preprocessor + model) is logged to MLflow.
7.  **Model Registration**: The new model version is registered in the MLflow Model Registry and assigned the `candidate` alias.
</details>

<details>
<summary><strong>Model Promotion (<code>scripts/promote_model.py</code>)</strong></summary>

This is a simple utility script that performs a critical governance step:
- It finds the model version currently aliased as `candidate`.
- It re-assigns its alias to `champion`, effectively promoting it to production.
- This is intended to be a manual step after a data scientist has verified the candidate model's quality.
</details>

<details>
<summary><strong>FastAPI Serving & Model Polling Strategy</strong></summary>

The FastAPI service is designed for a production environment where models are updated without downtime.
- **Initial Load**: On startup, the service loads the current `champion` model from MLflow using its model URI (`models:/{model_name}@champion`).
- **Periodic Polling**: A background task runs every 60 seconds to check if the `champion` alias in the MLflow Registry points to a new model version.
- **Atomic Swap**: If a new version is detected, the service downloads the new model and atomically replaces the model object in memory. This ensures that incoming prediction requests are always handled by a fully-loaded model, preventing race conditions or downtime.
- **Environment Variables**: The service requires `MLFLOW_TRACKING_URI` to connect to the MLflow server. `REGISTERED_MODEL_NAME` and `CHAMPION_ALIAS` can also be set to override defaults.
</details>

<details>
<summary><strong>Important Design Patterns</strong></summary>

- **Temporal Splits**: Data is split by date (`TRAIN_END_DATE`, `VAL_END_DATE` in `src/config.py`) to prevent data leakage and simulate a real-world production scenario where we train on the past to predict the future.
- **Bundled Preprocessor**: The scikit-learn `ColumnTransformer` (preprocessor) is included as the first step in the `sklearn.pipeline.Pipeline` that gets logged to MLflow. This ensures that the exact same transformations are applied during inference as during training.
- **Custom Thresholding**: The `ThresholdClassifier` is a custom wrapper that allows the `predict()` method to return class labels (0/1) based on a tuned probability threshold, which is essential for classification problems with imbalanced classes.
</end{details>

## 🛠️ Technology Stack

*   **Orchestration**: Docker, Docker Compose
*   **ML Lifecycle**: MLflow
*   **Model Training**: PyTorch, skorch, scikit-learn
*   **Hyperparameter Tuning**: Optuna
*   **API Framework**: FastAPI
*   **Data Handling**: Pandas, NumPy
*   **Object Storage**: MinIO (as S3-compatible backend for MLflow)
*   **Database**: PostgreSQL (as backend for MLflow)
