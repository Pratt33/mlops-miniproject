# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

        # Load model using DagsHub-compatible method
        cls.new_model = cls.load_model_dagshub_compatible()

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @classmethod
    def load_model_dagshub_compatible(cls):
        """Load model with DagsHub MLflow compatibility."""
        try:
            # Method 1: Try standard MLflow model registry
            model_name = "my_model"
            latest_version = cls.get_latest_model_version(model_name)
            if latest_version:
                model_uri = f'models:/{model_name}/{latest_version}'
                model = mlflow.pyfunc.load_model(model_uri)
                print(f"Model loaded from registry: {model_uri}")
                return model
        except Exception as registry_error:
            print(f"Registry loading failed: {registry_error}")
        
        try:
            # Method 2: Load from latest run with model tag
            client = mlflow.MlflowClient()
            experiment = mlflow.get_experiment_by_name("dvc-pipeline")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.model_name = 'my_model'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if runs:
                    run_id = runs[0].info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.pyfunc.load_model(model_uri)
                    print(f"Model loaded from tagged run: {model_uri}")
                    return model
        except Exception as tag_error:
            print(f"Tagged run loading failed: {tag_error}")
        
        try:
            # Method 3: Load from experiment_info.json (local fallback)
            with open('reports/experiment_info.json', 'r') as f:
                model_info = json.load(f)
            
            if 'run_id' in model_info:
                run_id = model_info['run_id']
                # Try to load from MLflow first
                try:
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.pyfunc.load_model(model_uri)
                    print(f"Model loaded from experiment_info run: {model_uri}")
                    return model
                except:
                    # Fallback to local pickle file
                    local_model = pickle.load(open('./models/model.pkl', 'rb'))
                    print("Model loaded from local pickle file")
                    return LocalModelWrapper(local_model)
                    
        except Exception as local_error:
            print(f"Local loading failed: {local_error}")
        
        # Final fallback: Load directly from local file
        local_model = pickle.load(open('./models/model.pkl', 'rb'))
        print("Model loaded from direct local pickle file")
        return LocalModelWrapper(local_model)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        try:
            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=[stage])
            return latest_version[0].version if latest_version else None
        except Exception:
            return None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

        # Print performance metrics for debugging
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy_new:.4f}")
        print(f"Precision: {precision_new:.4f}")
        print(f"Recall: {recall_new:.4f}")
        print(f"F1 Score: {f1_new:.4f}")


class LocalModelWrapper:
    """Wrapper to make local sklearn model compatible with MLflow pyfunc interface."""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        """Predict method compatible with MLflow pyfunc interface."""
        if hasattr(X, 'values'):  # pandas DataFrame
            return self.model.predict(X.values)
        else:  # numpy array
            return self.model.predict(X)


if __name__ == "__main__":
    unittest.main()