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

### Local inference

```bash
python llm_inference/local_inference.py
```

### Distributed inference

```bash
python llm_inference/distributed_inference.py
```
