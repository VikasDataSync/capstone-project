import numpy as np
import pandas as pd
import pickle
import json
import os
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn

try:
    from src.logger import logging
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if "src" in sys.modules:
        del sys.modules["src"]
    from src.logger import logging

ROOT_DIR = Path(__file__).resolve().parents[2]


def configure_mlflow_tracking() -> bool:
    """Configure MLflow tracking against DagsHub when token is available."""
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        logging.warning(
            "CAPSTONE_TEST not set. Skipping remote MLflow logging."
        )
        return False

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "VikasDataSync"
    repo_name = "capstone-project"
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    return True



def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(
    run_id: str | None,
    model_uri: str | None,
    file_path: str,
    mlflow_logged: bool,
) -> None:
    """Save the model run ID and URI to a JSON file."""
    try:
        model_info = {
            "run_id": run_id,
            "model_uri": model_uri,
            "mlflow_logged": mlflow_logged,
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    try:
        clf = load_model(str(ROOT_DIR / "models" / "model.pkl"))
        test_data = load_data(str(ROOT_DIR / "data" / "processed" / "test_bow.csv"))

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, str(ROOT_DIR / "reports" / "metrics.json"))

        mlflow_logged = False
        run_id = None
        model_uri = None
        if configure_mlflow_tracking():
            try:
                mlflow.set_experiment("my-dvc-pipeline")
                with mlflow.start_run() as run:
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    if hasattr(clf, "get_params"):
                        params = clf.get_params()
                        for param_name, param_value in params.items():
                            mlflow.log_param(param_name, param_value)

                    logged_model = mlflow.sklearn.log_model(clf, "model")
                    mlflow.log_artifact(str(ROOT_DIR / "reports" / "metrics.json"))
                    run_id = run.info.run_id
                    model_uri = logged_model.model_uri
                    mlflow_logged = True
            except Exception as e:
                logging.warning(
                    "MLflow logging skipped due to tracking server error: %s",
                    e,
                )

        save_model_info(
            run_id,
            model_uri,
            str(ROOT_DIR / "reports" / "experiment_info.json"),
            mlflow_logged,
        )
    except Exception as e:
        logging.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()
