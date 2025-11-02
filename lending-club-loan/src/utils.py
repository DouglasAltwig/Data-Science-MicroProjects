import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec

def set_model_alias(model_name, alias, run_id):
    """Sets an alias for the model version created by a specific run."""
    print(f"\n--- Setting '{alias}' alias for model from run {run_id} ---")
    client = MlflowClient()
    try:
        results = client.search_model_versions(f"name='{model_name}' AND run_id='{run_id}'")
        if not results:
            print(f"Warning: No model version found for '{model_name}' from run '{run_id}'. Alias not set.")
            return

        latest_version = results[0].version
        client.set_registered_model_alias(model_name, alias, latest_version)
        print(f"Alias '{alias}' set for version {latest_version} of model '{model_name}'.")
    except Exception as e:
        print(f"Error setting alias for model '{model_name}': {e}")

def create_model_signature(input_df: pd.DataFrame) -> ModelSignature:
    """Creates an MLflow ModelSignature for the given input DataFrame."""
    input_schema_list = []
    for col, dtype in input_df.dtypes.items():
        if np.issubdtype(dtype, np.floating):
            mlflow_type = "double"
        elif np.issubdtype(dtype, np.integer):
            mlflow_type = "long"
        else:
            mlflow_type = "string"
        input_schema_list.append(ColSpec(type=mlflow_type, name=col))
    
    input_schema = Schema(input_schema_list)
    output_schema = Schema([TensorSpec(np.dtype(np.int64), (-1,), "prediction")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)