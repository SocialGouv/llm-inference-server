import asyncio
import os

import torch
from fastapi import FastAPI
from transformers import pipeline

from llm_inference.generated_text_response import GeneratedTextResponse
from llm_inference.prompt_request import PromptRequest

app = FastAPI()


# Login to Hugging Face using the auth token
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if hf_token is None:
    raise ValueError(
        "Please set your Hugging Face token in the HUGGING_FACE_TOKEN environment variable."
    )

# Load the pipeline at startup
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

# Check for the best available device: NVIDIA GPU, MPS, or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU (M1, M2, M3, etc.)
else:
    device = torch.device("cpu")  # Fallback to CPU

text_generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float16,
    device=device,
    token=hf_token,
    return_full_text=False,
)


@app.post("/generate", response_model=GeneratedTextResponse)
async def generate_text(request: PromptRequest):
    with torch.no_grad():
        generated_texts = await asyncio.to_thread(
            text_generator,
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
    import uvicorn

    uvicorn.run("hf_inference_server:app", host="0.0.0.0", port=8000)
