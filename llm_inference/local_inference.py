import asyncio
import os
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Login to Hugging Face using the auth token
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if hf_token is None:
    raise ValueError(
        "Please set your Hugging Face token in the HUGGING_FACE_TOKEN environment variable."
    )

# Load the model and tokenizer at startup
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, token=hf_token, padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, token=hf_token
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define the request and response models
class PromptRequest(BaseModel):
    prompts: List[str]
    max_length: Optional[int] = 128
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_return_sequences: Optional[int] = 1
    do_sample: Optional[bool] = False


class GeneratedTextResponse(BaseModel):
    generated_texts: List[str]


@app.post("/generate", response_model=GeneratedTextResponse)
async def generate_text(request: PromptRequest):
    inputs = tokenizer(request.prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = await asyncio.to_thread(
            model.generate,
            **inputs,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_return_sequences=request.num_return_sequences,
            do_sample=request.do_sample,
        )

    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    return GeneratedTextResponse(generated_texts=generated_texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
