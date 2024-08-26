import asyncio
import os

import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_inference.generated_text_response import GeneratedTextResponse
from llm_inference.prompt_request import PromptRequest

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
device = next(model.parameters()).device
model.to(device)


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
