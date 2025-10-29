# Lending Club Loan Default Prediction Project

This project builds a deep neural network using PyTorch to predict loan defaults based on the Lending Club dataset. It includes a complete MLOps pipeline featuring:

- **Data Cleaning and Feature Engineering**: Robust preparation of the raw dataset.
- **Hyperparameter Tuning**: Using Optuna to find the best model architecture and training parameters, with the ability to resume long-running experiments.
- **MLflow Integration**: Tracking experiments, logging metrics, artifacts (like confusion matrices), and managing model versions.
- **GPU Support**: The PyTorch model will automatically use a CUDA-enabled GPU if available.

## Project Structure

- `main.py`: The main script to run the entire pipeline.
- `data_loader.py`: Handles data loading and initial cleaning.
- `preprocess.py`: Feature engineering and preprocessing pipeline.
- `models.py`: PyTorch DNN model definition.
- `evaluate.py`: Final model evaluation and visualization functions.
- `LendingClubLoan.ipynb`: Notebook containing the in-depth Loan Dataset analysis.

---

### Getting Started: Lending Club Loan Project

This guide will walk you through cloning the project repository, starting the required services with Docker Compose, and accessing the Jupyter Lab environment.

---

### Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Git:** For cloning the repository.
2.  **Docker and Docker Compose:** To build and run the containerized application. (Most modern Docker Desktop installations include Docker Compose).
3.  **NVIDIA GPU and Drivers:** The environment is configured to use an NVIDIA GPU.
4.  **NVIDIA Container Toolkit:** This is essential for Docker to be able to access your GPU. You can find installation instructions on the [NVIDIA GitHub repository](https://github.com/NVIDIA/nvidia-docker).

---

### Step 1: Clone the GitHub Repository

First, you need to download the project files from GitHub. Open your terminal or command prompt and run the following command. This will create a new directory named `Data-Science-MicroProjects` containing all the project files.

```bash
git clone https://github.com/DouglasAltwig/Data-Science-MicroProjects.git
```

### Step 2: Change Into the Project Directory

The repository contains multiple projects. You need to navigate specifically into the `lending-club-loan` sub-directory, which contains the `docker-compose.yml` file for this project.

```bash
cd Data-Science-MicroProjects/lending-club-loan
```
**Important:** All subsequent commands must be run from within this directory.

### Step 3: Spin Up the Environment with Docker Compose

Now you will use Docker Compose to build the image (if needed) and start the container. The `-d` flag runs the container in "detached" mode, meaning it will run in the background and not tie up your terminal.

Execute this command from the `lending-club-loan` directory:

```bash
docker-compose up -d
```
*(Note: If you are using a newer version of Docker, the command may be `docker compose up -d` without the hyphen).*

**What happens now?**
*   Docker will check if you have the `nvcr.io/nvidia/rapidsai/notebooks:25.08-cuda12.0-py3.12` image locally.
*   If not, it will download the image. This can take several minutes depending on your internet connection, as the image is quite large.
*   Once the image is ready, Docker Compose will create and start a container named `lending_club_loan_container` based on the configuration in your `docker-compose.yml` file.

You can check if the container is running with the command `docker ps`.

### Step 4: Access the Jupyter Lab Service

The container is now running and has exposed port `8888` to your local machine. You can access the Jupyter Lab environment through your web browser.

1.  Open your favorite web browser (e.g., Chrome, Firefox, Safari).
2.  Navigate to the following address:
    ```
    http://localhost:8888
    ```
    or
    ```
    http://127.0.0.1:8888
    ```
3.  You should see the Jupyter Lab interface load directly, without asking for a password or token.

Inside Jupyter Lab, you will see a file browser on the left. Look for a folder named **`extra`**. This folder is directly mapped to the project files on your computer, so any changes you make to notebooks or files inside this `extra` folder will be saved directly to your machine.

---

### How to Stop the Environment

When you are finished working, you can stop the container by returning to the same terminal window (in the `lending-club-loan` directory) and running:

```bash
docker-compose down
```

This command will gracefully stop and remove the container and its associated network. Your work will remain safe in the project directory.