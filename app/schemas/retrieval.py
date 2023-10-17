from typing import Optional

from pydantic import BaseModel


class ChatBotParameters(BaseModel):
    vectorstore: Optional[str] = "chromadb"
    embedding_model: Optional[str] = "openai"
    temperature: Optional[float] = 0
    prompt: str

    class Config:
        json_schema_extra = {
            "example": {
                "vectorstore": "chromadb",
                "embedding_model": "openai",
                "temperature": 0,
                "prompt": "What is the API path for the risks section in the Teamwork API?"
            }
        }
