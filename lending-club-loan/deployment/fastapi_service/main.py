import os
import sys

# Add project root to Python path
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

import asyncio
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv


class LoanApplication(BaseModel):
    loan_amnt: float = Field(default=25000.0, description="The listed amount of the loan applied for by the borrower.")
    term: str = Field(default="36 months", description="The number of payments on the loan. Values are in months and can be either 36 or 60.")
    int_rate: float = Field(default=13.56)
    installment: float = Field(default=849.82)
    sub_grade: str = Field(default="C1")
    emp_length: float = Field(default=10.0, description="Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.")
    home_ownership: str = Field(default="RENT")
    annual_inc: float = Field(default=55000.0)
    verification_status: str = Field(default="Verified")
    pymnt_plan: str = Field(default="n")
    purpose: str = Field(default="debt_consolidation")
    addr_state: str = Field(default="CA")
    dti: float = Field(default=23.95)
    delinq_2yrs: float = Field(default=0.0)
    fico_range_low: float = Field(default=695.0)
    fico_range_high: float = Field(default=699.0)
    inq_last_6mths: float = Field(default=0.0)
    open_acc: float = Field(default=10.0)
    pub_rec: float = Field(default=0.0)
    revol_bal: float = Field(default=11602.0)
    revol_util: float = Field(default=65.2)
    total_acc: float = Field(default=24.0)
    initial_list_status: str = Field(default="w")
    collections_12_mths_ex_med: float = Field(default=0.0)
    application_type: str = Field(default="Individual")
    acc_now_delinq: float = Field(default=0.0)
    tot_coll_amt: float = Field(default=0.0)
    tot_cur_bal: float = Field(default=150000.0)
    total_rev_hi_lim: float = Field(default=30000.0)
    acc_open_past_24mths: float = Field(default=4.0)
    avg_cur_bal: float = Field(default=15000.0)
    bc_open_to_buy: float = Field(default=5000.0)
    bc_util: float = Field(default=75.0)
    chargeoff_within_12_mths: float = Field(default=0.0)
    delinq_amnt: float = Field(default=0.0)
    mo_sin_old_il_acct: float = Field(default=120.0)
    mo_sin_old_rev_tl_op: float = Field(default=180.0)
    mo_sin_rcnt_rev_tl_op: float = Field(default=12.0)
    mo_sin_rcnt_tl: float = Field(default=6.0)
    mort_acc: float = Field(default=1.0)
    mths_since_recent_bc: float = Field(default=24.0)
    mths_since_recent_inq: float = Field(default=6.0)
    num_accts_ever_120_pd: float = Field(default=0.0)
    num_actv_bc_tl: float = Field(default=3.0)
    num_actv_rev_tl: float = Field(default=5.0)
    num_bc_sats: float = Field(default=3.0)
    num_bc_tl: float = Field(default=5.0)
    num_il_tl: float = Field(default=10.0)
    num_op_rev_tl: float = Field(default=8.0)
    num_rev_accts: float = Field(default=12.0)
    num_rev_tl_bal_gt_0: float = Field(default=5.0)
    num_sats: float = Field(default=10.0)
    num_tl_120dpd_2m: float = Field(default=0.0)
    num_tl_30dpd: float = Field(default=0.0)
    num_tl_90g_dpd_24m: float = Field(default=0.0)
    num_tl_op_past_12m: float = Field(default=2.0)
    pct_tl_nvr_dlq: float = Field(default=95.0)
    percent_bc_gt_75: float = Field(default=50.0)
    pub_rec_bankruptcies: float = Field(default=0.0)
    tax_liens: float = Field(default=0.0)
    tot_hi_cred_lim: float = Field(default=200000.0)
    total_bal_ex_mort: float = Field(default=40000.0)
    total_bc_limit: float = Field(default=15000.0)
    total_il_high_credit_limit: float = Field(default=50000.0)
    hardship_flag: str = Field(default="N")
    disbursement_method: str = Field(default="Cash")
    debt_settlement_flag: str = Field(default="N")
    credit_history_length: float = Field(default=15.0)
    issue_year: int = Field(default=2018)
    issue_month: int = Field(default=10)
    loan_amnt_div_instlmnt: float = Field(default=29.42)
    loan_to_income_ratio: float = Field(default=0.45)
    dti_x_income: float = Field(default=1317250.0)
    revol_util_x_inq: float = Field(default=0.0)

class PredictionResponse(BaseModel):
    prediction: int
    label: str

model_state = {
    "model": None,
    "current_version": None,
    "model_name": os.getenv("REGISTERED_MODEL_NAME", "lending_club_loan_classification_model"),
    "model_alias": os.getenv("CHAMPION_ALIAS", "champion"),
}

def load_champion_model():
    """Loads the champion model from MLflow registry."""
    model_uri = f"models:/{model_state['model_name']}@{model_state['model_alias']}"
    try:
        client = mlflow.tracking.MlflowClient()
        latest_version_obj = client.get_model_version_by_alias(
            model_state['model_name'], model_state['model_alias']
        )
        latest_version = latest_version_obj.version

        if latest_version != model_state["current_version"]:
            print(f"New champion model version detected: {latest_version}. (Previously: {model_state['current_version']})")
            model_state["model"] = mlflow.pyfunc.load_model(model_uri)
            model_state["current_version"] = latest_version
            print(f"Successfully loaded model version {latest_version}.")
        else:
            print(f"Model version {latest_version} is already loaded.")
            
    except Exception as e:
        print(f"Error loading model: {e}. API will continue with old model if available.")
        if model_state["model"] is None:
             model_state["current_version"] = None


async def model_polling_task():
    """A background task that periodically checks for a new model."""
    while True:
        try:
            load_champion_model()
        except Exception as e:
            print(f"Error during model polling: {e}")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    load_dotenv()
    
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("MLFLOW_TRACKING_URI environment variable not set.")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")
    
    print("Performing initial model load...")
    load_champion_model()
    
    print("Starting background model polling task...")
    asyncio.create_task(model_polling_task())
    
    yield
    
    # --- Shutdown ---
    model_state.clear()
    print("Application shut down.")

app = FastAPI(
    title="Loan Default Prediction API",
    description="Serves a PyTorch model via skorch and MLflow for predicting loan defaults. The model is automatically updated from the MLflow registry.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Loan Default Prediction API"}

@app.get("/health", tags=["General"])
def health_check():
    """Check if the API is running and the model is loaded."""
    if model_state.get("model") is not None:
        return {
            "status": "ok", 
            "model_loaded": True,
            "model_name": model_state["model_name"],
            "model_alias": model_state["model_alias"],
            "current_version": model_state["current_version"],
        }
    return {"status": "error", "model_loaded": False}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: LoanApplication, request: Request):
    """
    Predict loan default for a single application.
    - **Prediction 0**: Good Loan
    - **Prediction 1**: Bad Loan (Default)
    """
    model = model_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Please check the server logs."
        )

    try:
        input_df = pd.DataFrame([application.model_dump()])
        prediction_result = model.predict(input_df)
        prediction_int = int(prediction_result[0])
        label = "Bad Loan" if prediction_int == 1 else "Good Loan"
        
        return {"prediction": prediction_int, "label": label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")