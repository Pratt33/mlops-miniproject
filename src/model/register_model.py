# register model

import json
import mlflow
import logging
import os
import dagshub

dagshub.init(repo_owner='Pratt33', repo_name='mlops-miniproject', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Pratt33/mlops-miniproject.mlflow")


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model_dagshub_compatible(model_name: str, model_info: dict):
    """Register the model to MLflow with DagsHub compatibility."""
    try:
        # Try standard MLflow model registry first
        model_uri = f"runs:/{model_info['run_id']}/model"
        
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Try to transition to staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        return True
        
    except Exception as registry_error:
        logger.warning(f"Standard model registry failed: {registry_error}")
        
        # Fallback: Create model registry manually using tags and artifacts
        try:
            with mlflow.start_run(run_id=model_info['run_id']) as run:
                # Add model registry tags as metadata
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("model_stage", "Staging")
                mlflow.set_tag("registered_model", "true")
                mlflow.set_tag("model_version", "1")  # Simple versioning
                
                # Log model metadata
                import time
                model_registry_info = {
                    "model_name": model_name,
                    "stage": "Staging",
                    "run_id": model_info['run_id'],
                    "model_path": "model",
                    "registered_at": int(time.time() * 1000),  # Current timestamp in milliseconds
                    "status": "registered_via_fallback"
                }
                
                # Save registry info as artifact
                registry_file = "model_registry_info.json"
                with open(registry_file, 'w') as f:
                    json.dump(model_registry_info, f, indent=4)
                
                mlflow.log_artifact(registry_file, "model_registry")
                os.remove(registry_file)  # Clean up
                
                logger.debug(f'Model {model_name} registered using fallback method (tags + artifacts).')
                return True
                
        except Exception as fallback_error:
            logger.error(f"Fallback registration also failed: {fallback_error}")
            return False

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        success = register_model_dagshub_compatible(model_name, model_info)
        
        if success:
            print(f"Model {model_name} registered successfully!")
        else:
            print(f"Failed to register model {model_name}")
            
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()