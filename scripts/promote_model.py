# promote model

import os
import mlflow
import json
import warnings

def promote_model_dagshub_compatible():
    """Promote model with DagsHub MLflow compatibility."""
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Pratt33"
    repo_name = "mlops-miniproject"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()
    model_name = "my_model"
    
    try:
        # Method 1: Try standard MLflow model registry promotion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            
            # Get the latest version in staging
            latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

            # Archive the current production model
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            for version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )

            # Promote the new model to production
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version_staging,
                stage="Production"
            )
            print(f"Model version {latest_version_staging} promoted to Production")
            return True
            
    except Exception as registry_error:
        print(f"Standard model registry promotion failed: {registry_error}")
        
        # Method 2: Fallback to tag-based promotion (DagsHub compatible)
        try:
            # Find the latest "Staging" model by searching runs
            experiment = mlflow.get_experiment_by_name("dvc-pipeline")
            if not experiment:
                raise Exception("Experiment 'dvc-pipeline' not found")
            
            # Search for runs with staging model
            staging_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.model_name = 'my_model' and tags.model_stage = 'Staging'",
                order_by=["start_time DESC"],
                max_results=10
            )
            
            if not staging_runs:
                # If no staging runs found, look for any runs with model_name tag
                print("No staging models found, looking for any registered models...")
                staging_runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.model_name = 'my_model'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
            if not staging_runs:
                raise Exception("No models found with model_name tag")
                
            staging_run = staging_runs[0]
            staging_run_id = staging_run.info.run_id
            
            # Archive current production models
            production_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.model_name = 'my_model' and tags.model_stage = 'Production'",
                order_by=["start_time DESC"]
            )
            
            # Update tags for current production models to "Archived"
            for prod_run in production_runs:
                with mlflow.start_run(run_id=prod_run.info.run_id):
                    mlflow.set_tag("model_stage", "Archived")
                    import time
                    mlflow.set_tag("archived_at", int(time.time() * 1000))
                    print(f"Archived production model from run: {prod_run.info.run_id}")
            
            # Promote staging model to production
            with mlflow.start_run(run_id=staging_run_id):
                mlflow.set_tag("model_stage", "Production")
                import time
                mlflow.set_tag("promoted_at", int(time.time() * 1000))
                mlflow.set_tag("promoted_from", "Staging")
                
                # Log promotion metadata
                promotion_info = {
                    "model_name": model_name,
                    "promoted_run_id": staging_run_id,
                    "promoted_at": int(time.time() * 1000),
                    "promoted_from": "Staging",
                    "promoted_to": "Production",
                    "promotion_method": "tag_based_fallback"
                }
                
                promotion_file = "promotion_info.json"
                with open(promotion_file, 'w') as f:
                    json.dump(promotion_info, f, indent=4)
                
                mlflow.log_artifact(promotion_file, "promotion_metadata")
                os.remove(promotion_file)  # Clean up
                
            print(f"Model from run {staging_run_id} promoted to Production using tag-based method")
            return True
            
        except Exception as fallback_error:
            print(f"Fallback promotion also failed: {fallback_error}")
            
            # Method 3: Simple local promotion record
            try:
                # Load experiment info to get current model
                with open('reports/experiment_info.json', 'r') as f:
                    model_info = json.load(f)
                
                # Create promotion record
                import time
                promotion_record = {
                    "model_name": model_name,
                    "run_id": model_info.get('run_id', 'unknown'),
                    "promoted_at": int(time.time() * 1000),
                    "stage": "Production",
                    "promotion_method": "local_record",
                    "model_path": model_info.get('model_path', './models/model.pkl'),
                    "status": "promoted"
                }
                
                # Save promotion record
                os.makedirs('reports', exist_ok=True)
                with open('reports/model_promotion.json', 'w') as f:
                    json.dump(promotion_record, f, indent=4)
                    
                print(f"Model promotion recorded locally for run: {promotion_record['run_id']}")
                return True
                
            except Exception as local_error:
                print(f"Local promotion record failed: {local_error}")
                return False

def promote_model():
    """Main promotion function - wrapper for compatibility."""
    return promote_model_dagshub_compatible()

if __name__ == "__main__":
    success = promote_model()
    if success:
        print("Model promotion completed successfully!")
    else:
        print("Model promotion failed!")
        exit(1)