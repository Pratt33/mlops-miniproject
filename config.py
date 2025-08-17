# Configuration file for MLOps project
import os

# MLflow tracking configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'https://dagshub.com/Pratt33/mlops-miniproject.mlflow')

# DagHub configuration
DAGSHUB_REPO_OWNER = os.getenv('DAGSHUB_REPO_OWNER', 'Pratt33')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME', 'mlops-miniproject')

# Model configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'sentiment_model')

# Data paths
DATA_RAW_PATH = 'data/raw'
DATA_PROCESSED_PATH = 'data/processed'
DATA_INTERIM_PATH = 'data/interim'

# Model paths
MODELS_PATH = 'models'
MODEL_FILE = 'model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'

# Reports paths
REPORTS_PATH = 'reports'
METRICS_FILE = 'metrics.json'
