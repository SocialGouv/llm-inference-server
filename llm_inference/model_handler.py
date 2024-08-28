import os

import boto3
import torch
from tqdm import tqdm
from transformers import pipeline


class S3ModelHandler:
    def __init__(
        self,
        bucket_name: str,
        local_model_dir: str,
        s3_model_path: str,
        s3_endpoint_url: str,
        s3_access_key_id: str,
        s3_secret_access_key: str,
    ):
        self.bucket_name = bucket_name
        self.local_model_dir = local_model_dir
        self.s3_model_path = s3_model_path
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_secret_access_key = s3_secret_access_key
        self.s3_access_key_id = s3_access_key_id
        self.text_generator = None

    def download_model_from_s3(self):
        print(
            f"Downloading model {self.s3_model_path} from S3 bucket {self.bucket_name}..."
        )
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.s3_access_key_id,
            aws_secret_access_key=self.s3_secret_access_key,
            endpoint_url=self.s3_endpoint_url,
        )

        # Download the model files
        if not os.path.exists(self.local_model_dir):
            os.makedirs(self.local_model_dir)

        # List all objects in the S3 folder
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket_name, Prefix=self.s3_model_path
        ):
            for obj in page.get("Contents", []):
                # Check if the object is a file (not a folder)
                if not obj["Key"].endswith("/"):
                    # Define the local path for the object
                    local_file_path = os.path.join(
                        self.local_model_dir,
                        os.path.relpath(obj["Key"], self.s3_model_path),
                    )

                    # Create the necessary local subdirectories if they don't exist
                    if not os.path.exists(os.path.dirname(local_file_path)):
                        os.makedirs(os.path.dirname(local_file_path))

                    # Get the file size
                    response = s3_client.head_object(
                        Bucket=self.bucket_name, Key=obj["Key"]
                    )
                    file_size = int(response["ContentLength"])

                    # Create a progress bar
                    progress_bar = tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        desc=f'Downloading {obj["Key"]}',
                    )

                    # Download the file
                    s3_client.download_file(
                        self.bucket_name,
                        obj["Key"],
                        local_file_path,
                        Callback=progress_bar.update,
                    )

                    # Close the progress bar
                    progress_bar.close()
        print("Model downloaded successfully!")

    def load_model(self):
        """Load the model into memory"""
        print("Loading model...")

        model_name_or_path = self.local_model_dir
        # Initialize the model with model parallelism
        device_map = "auto"  # Automatically split across available GPUs
        self.text_generator = pipeline(
            "text-generation",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            torch_dtype=torch.bfloat16
            if torch.backends.mps.is_available()
            else torch.float16,
            device_map=device_map,
            return_full_text=False,
        )
        print("Model loaded successfully!")
