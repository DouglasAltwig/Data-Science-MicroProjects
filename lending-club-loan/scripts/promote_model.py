import os
import argparse
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Add project root to path to allow absolute imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def promote_candidate_to_champion(model_name: str, candidate_alias: str, champion_alias: str):
    """
    Promotes the model version with the 'candidate' alias to 'champion'.
    """
    load_dotenv() # Load environment variables from .env file for local execution
    
    # MLFLOW_TRACKING_URI is expected to be set in the environment (e.g., in docker-compose)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable not set.")
        
    print(f"Connecting to MLflow at: {tracking_uri}")
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        # 1. Find the version number of the current candidate
        candidate_version = client.get_model_version_by_alias(model_name, candidate_alias)
        version_number = candidate_version.version
        run_id = candidate_version.run_id
        
        print(f"Found candidate model: '{model_name}' version {version_number} from run {run_id}.")

        # 2. Promote this version to be the new champion
        print(f"Promoting version {version_number} to alias '{champion_alias}'...")
        client.set_registered_model_alias(model_name, champion_alias, version_number)

        print("\n--- Promotion Successful ---")
        print(f"Model: '{model_name}'")
        print(f"Version: {version_number}")
        print(f"Now has the '{champion_alias}' alias and is ready for production.")
        
        # Optional: Remove the candidate alias after promotion
        print(f"Removing '{candidate_alias}' alias from version {version_number}.")
        client.delete_registered_model_alias(model_name, candidate_alias)

    except Exception as e:
        print(f"Error during promotion: {e}")
        print(f"It's possible no model version has the '{candidate_alias}' alias yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a model from candidate to champion.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=config.REGISTERED_MODEL_NAME,
        help="Name of the registered model in MLflow."
    )
    args = parser.parse_args()

    promote_candidate_to_champion(
        model_name=args.model_name,
        candidate_alias=config.CANDIDATE_ALIAS,
        champion_alias=config.CHAMPION_ALIAS
    )