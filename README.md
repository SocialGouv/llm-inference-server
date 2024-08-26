# LLM Inference

## Setup

```bash
poetry install
poetry shell
```

### Env variables

```bash
export HUGGING_FACE_TOKEN=<hugging_face_token>
export S3_ACCESS_KEY_ID=<s3_access_key_id>
export S3_SECRET_ACCESS_KEY=<s3_secret_access_key>
export S3_ENDPOINT_URL="https://s3.gra.io.cloud.ovh.net"
```

## Run

### Huggingface inference server

```bash
python llm_inference/hf_inference_server.py
```

### S3 inference server

```bash
python llm_inference/s3_inference_server.py
```

## Test

### Batch requests

```bash
python scripts/example_batch_request.py
```
