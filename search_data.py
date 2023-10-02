import boto3
import pickle
import pandas as pd
from io import StringIO


class S3:
    def __init__(self, role_arn):
        """S3 Client Wrapper
        Args:
            role_arn (str): S3 role
        """
        self.s3 = None
        self.connect(arn=role_arn)

        self.bucket_name = None
        self.bucket = None

    def connect(self, arn):
        """connect and set boto S3 object

        Raises:
            Exception: Access Denied
        """
        general_sts_client = boto3.client("sts")
        response = general_sts_client.assume_role(
            RoleArn=arn,
            RoleSessionName="AssumeRoleSession1"
        )

        credentials = response["Credentials"]
        self.s3 = boto3.resource(
            "s3",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"]
        )

    def connect_with_keys(self, access_key: str, secret_key: str):
        self.s3 = boto3.resource(
            "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )

    def set_bucket(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.bucket = self.s3.Bucket(self.bucket_name)

    def get_folder_content(self, folder_name: str):
        self._assert_bucket()

        return self.bucket.objects.filter(Prefix=folder_name)

    def download_file(self, source_path: str, destination_path: str):
        """Download a file from S3 - source path to destination path
        Args:
            source_path (str): S3 filepath
            destination_path (str): local directory path with file name
        """
        self._assert_bucket()

        self.bucket.download_file(source_path, destination_path)

    def upload_file(self, source_path: str, destination_path: str):
        """Upload a file from S3 - source path to destination path
        Args:
            source_path (str): local directory path
            destination_path (str): S3 filepath
        """
        self._assert_bucket()

        self.s3.meta.client.upload_file(source_path, self.bucket_name, destination_path)

    def copy_file(self, source_path: str, destination_path: str):
        """Upload a file from S3 - source path to destination path
        Args:
            source_path (str): local directory path
            destination_path (str): S3 filepath
        """
        self._assert_bucket()

        self.s3.meta.client.copy(source_path, self.bucket_name, destination_path)

    def load_df(self, source_path: str, **kwargs):
        """Load file from s3 as pandas dataframe
        Args:
            source_path (str): S3 filepath
        Returns:
            _type_: pd.DataFrame
        """
        self._assert_bucket()

        csv_obj = self.s3.Object(self.bucket_name, source_path).get()
        body = csv_obj["Body"]
        csv_string = body.read().decode("utf-8")
        return pd.read_csv(StringIO(csv_string), **kwargs)

    def load_df_from_pickle(self, key_path):
        response = self.s3.Bucket(self.bucket_name).Object(key_path).get()
        body_string = response['Body'].read()

        return pickle.loads(body_string)

    def _assert_bucket(self):
        assert self.bucket is not None, "bucket name cannot be None"


if __name__ == "__main__":
    from datetime import datetime, timedelta

    # iam role
    arn = ""
    
    # s3 bucket name (production)
    bucket = "pe-mum-word-corpus-production"
    
    # s3 file path
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    s3_path = f"corpus_history/case/v3/{yesterday}/raw_data.pkl"

    # local path to save the downloaded file
    destination_path = "./raw_data.pkl"

    # connect
    s3 = S3(role_arn=arn)

    # download production data
    s3.set_bucket(bucket_name=bucket)
    s3.download_file(source_path=s3_path, destination_path=destination_path)

    # read the downloaded data
    data = pickle.load(open(destination_path, "rb"))

    """
    This is the production search data
    Query/use it as per need
    """