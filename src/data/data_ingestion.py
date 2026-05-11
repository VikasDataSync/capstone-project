import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

try:
    from src.connections import s3_connection
    from src.logger import logging
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if "src" in sys.modules:
        del sys.modules["src"]
    from src.connections import s3_connection
    from src.logger import logging

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params(params_path: Path) -> dict:
    """Load parameters from a YAML file."""
    try:
        with params_path.open("r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logging.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML error: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logging.error(
            "Unexpected error occurred while loading the data: %s",
            e,
        )
        raise


def get_ingestion_params() -> dict:
    params = load_params(params_path=DEFAULT_PARAMS_PATH)
    if "data_ingestion" not in params:
        raise KeyError("Missing 'data_ingestion' section in params.yaml")
    return params["data_ingestion"]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        logging.info("pre-processing...")
        final_df = df[df["sentiment"].isin(["positive", "negative"])].copy()
        final_df["sentiment"] = final_df["sentiment"].map(
            {"positive": 1, "negative": 0}
        ).astype("int64")
        logging.info("Data preprocessing completed")
        return final_df
    except KeyError as e:
        logging.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(
            os.path.join(raw_data_path, "train.csv"),
            index=False,
        )
        test_data.to_csv(
            os.path.join(raw_data_path, "test.csv"),
            index=False,
        )
        logging.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logging.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main():
    try:
        ingestion_params = get_ingestion_params()
        test_size = ingestion_params["test_size"]
        if not 0 < test_size < 1:
            raise ValueError(
                "data_ingestion.test_size must be between 0 and 1."
            )

        s3 = s3_connection.s3_operations(
            bucket_name=ingestion_params["bucket_name"],
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=ingestion_params.get(
                "aws_region",
                "us-east-1",
            ),
        )
        s3_file_key = ingestion_params.get("s3_file_key", "data.csv")
        df = s3.fetch_file_from_s3(s3_file_key)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )
        save_data(
            train_data,
            test_data,
            data_path=str(
                ROOT_DIR / ingestion_params.get("output_data_dir", "data")
            ),
        )
    except Exception as e:
        logging.error(
            "Failed to complete the data ingestion process: %s",
            e,
        )
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
