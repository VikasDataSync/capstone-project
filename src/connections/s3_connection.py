import boto3
import pandas as pd
from io import StringIO

from src.logger import logging


class s3_operations:
    def __init__(
        self,
        bucket_name,
        aws_access_key=None,
        aws_secret_key=None,
        region_name="us-east-1",
    ):
        """
        Initialize with S3 bucket details and optional
        explicit AWS credentials.
        """
        self.bucket_name = bucket_name
        if bool(aws_access_key) ^ bool(aws_secret_key):
            raise ValueError(
                "Both aws_access_key and aws_secret_key must be "
                "provided together."
            )

        client_kwargs = {"region_name": region_name}
        if aws_access_key and aws_secret_key:
            client_kwargs["aws_access_key_id"] = aws_access_key
            client_kwargs["aws_secret_access_key"] = aws_secret_key

        self.s3_client = boto3.client("s3", **client_kwargs)
        logging.info("Data Ingestion from S3 bucket initialized")

    def fetch_file_from_s3(self, file_key):
        """
        Fetch a CSV file from S3 and return it as a Pandas DataFrame.
        :param file_key: S3 file path (e.g., 'data/data.csv')
        :return: Pandas DataFrame
        """
        try:
            logging.info(
                "Fetching file '%s' from S3 bucket '%s'...",
                file_key,
                self.bucket_name,
            )
            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key,
            )
            df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
            logging.info(
                "Successfully fetched '%s' from S3 with %s records.",
                file_key,
                len(df),
            )
            return df
        except Exception as e:
            logging.exception(f"❌ Failed to fetch '{file_key}' from S3: {e}")
            raise
