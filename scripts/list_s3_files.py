import os

import boto3

BUCKET_NAME = "vllm-cache"
s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")

# Initialize the S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=s3_access_key_id,
    aws_secret_access_key=s3_secret_access_key,
    endpoint_url=s3_endpoint_url,
)

# List objects in the S3 bucket to verify the path
response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)

for obj in response.get("Contents", []):
    print(obj["Key"])
