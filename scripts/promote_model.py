# promote model

import os
import mlflow
from mlflow.exceptions import MlflowException


def _search_latest_version(client, model_name, stage=None):
    versions = client.search_model_versions(f"name='{model_name}'")
    if stage is not None:
        versions = [
            version for version in versions
            if getattr(version, "current_stage", "").lower() == stage.lower()
        ]
    if not versions:
        return None
    return max(versions, key=lambda version: int(version.version)).version

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "VikasDataSync"
    repo_name = "capstone-project"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging. Fall back to search API where stage APIs are unavailable.
    try:
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        latest_version_staging = staging_versions[0].version if staging_versions else None
    except MlflowException:
        latest_version_staging = _search_latest_version(client, model_name, stage="Staging")

    if latest_version_staging is None:
        latest_version_staging = _search_latest_version(client, model_name)

    if latest_version_staging is None:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    # Archive the current production model
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    except MlflowException:
        prod_version = _search_latest_version(client, model_name, stage="Production")
        prod_versions = [type("Version", (), {"version": prod_version})()] if prod_version else []

    for version in prod_versions:
        if str(version.version) == str(latest_version_staging):
            continue
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

if __name__ == "__main__":
    promote_model()
