import asyncio
import os

import torch
import uvicorn
from fastapi import FastAPI

from llm_inference.generated_text_response import GeneratedTextResponse
from llm_inference.model_handler import S3ModelHandler
from llm_inference.prompt_request import PromptRequest

app = FastAPI()

# S3 configuration
BUCKET_NAME = "vllm-cache"
S3_MODEL_PATH = os.getenv("S3_MODEL_PATH")
local_model_dir = f"models/{S3_MODEL_PATH}"
s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")

# Check if S3 credentials are set
if not s3_access_key_id or not s3_secret_access_key:
    raise ValueError(
        "Please set your S3 credentials in the environment variables: S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_ENDPOINT_URL (See README.md)."
    )

model_handler = S3ModelHandler(
    bucket_name=BUCKET_NAME,
    local_model_dir=local_model_dir,
    s3_model_path=S3_MODEL_PATH,
    s3_endpoint_url=s3_endpoint_url,
    s3_access_key_id=s3_access_key_id,
    s3_secret_access_key=s3_secret_access_key,
)

try:
    model_handler.load_model()
except OSError:
    print("Loading the model failed.")
    model_handler.download_model_from_s3()
    model_handler.load_model()


@app.post("/generate", response_model=GeneratedTextResponse)
async def generate_text(request: PromptRequest):
    with torch.no_grad():
        generated_texts = await asyncio.to_thread(
            model_handler.text_generator,
            request.prompts,
            do_sample=request.do_sample,
            max_new_tokens=request.max_new_tokens,
            num_return_sequences=request.num_return_sequences,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

    # Extract the generated text from the pipeline output
    generated_texts = [result[0]["generated_text"] for result in generated_texts]

    return GeneratedTextResponse(generated_texts=generated_texts)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
