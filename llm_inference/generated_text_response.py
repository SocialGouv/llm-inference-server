from pydantic import BaseModel

class GeneratedTextResponse(BaseModel):
    generated_texts: list[str]
