# Define the request and response models
from typing import Optional

from pydantic import BaseModel


class PromptRequest(BaseModel):
    prompts: list[str]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_return_sequences: Optional[int] = 1
    do_sample: Optional[bool] = False
    max_new_tokens: Optional[int] = 64
