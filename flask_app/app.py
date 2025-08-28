from flask import Flask, render_template, request
from preprocessing import normalize_text
import joblib
import os
import mlflow
import pickle
import json

app = Flask(__name__)

def load_production_model():
    """Load the current production model with MLflow integration."""
    try:
        # Method 1: Try to load from MLflow (DagsHub compatible)
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            mlflow.set_tracking_uri("https://dagshub.com/Pratt33/mlops-miniproject.mlflow")
            
            # Look for production model using tags
            client = mlflow.MlflowClient()
            experiment = mlflow.get_experiment_by_name("dvc-pipeline")
            
            if experiment:
                # Search for production models
                production_runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.model_name = 'my_model' and tags.model_stage = 'Production'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if production_runs:
                    run_id = production_runs[0].info.run_id
                    try:
                        # Try to download model from MLflow
                        model_uri = f"runs:/{run_id}/model"
                        model = mlflow.pyfunc.load_model(model_uri)
                        print(f"Loaded production model from MLflow run: {run_id}")
                        return model, "mlflow"
                    except Exception as download_error:
                        print(f"Failed to download from MLflow: {download_error}")
                        
    except Exception as mlflow_error:
        print(f"MLflow loading failed: {mlflow_error}")
    
    # Method 2: Check local promotion record
    try:
        if os.path.exists('reports/model_promotion.json'):
            with open('reports/model_promotion.json', 'r') as f:
                promotion_info = json.load(f)
            if promotion_info.get('stage') == 'Production':
                print("Using locally promoted model")
    except Exception as promotion_error:
        print(f"Promotion record check failed: {promotion_error}")
    
    # Method 3: Fallback to local files (always works)
    try:
        model = joblib.load('models/model.pkl')
        print("Loaded model from local pickle file")
        return LocalModelWrapper(model), "local"
    except Exception as local_error:
        print(f"Local model loading failed: {local_error}")
        raise Exception("No model could be loaded from any source")

class LocalModelWrapper:
    """Wrapper to make local sklearn model work like MLflow pyfunc model."""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        """Predict method compatible with MLflow pyfunc interface."""
        if hasattr(X, 'values'):  # pandas DataFrame
            return self.model.predict(X.values)
        else:  # numpy array or sparse matrix
            return self.model.predict(X)

# Load model and vectorizer at startup
model, model_source = load_production_model()
vectorizer = joblib.load('models/vectorizer.pkl')
print(f"Flask app initialized with model from: {model_source}")

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = normalize_text(text)
    features = vectorizer.transform([text])
    result = model.predict(features)
    return render_template('index.html', result=result[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')