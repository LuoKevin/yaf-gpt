from pydantic import BaseModel
from pydantic.v1 import BaseSettings

class ChatConfig(BaseModel):
    system_prompt: str | None = None
    temperature: float  = 0.0
    max_tokens: int = 30

class Settings(BaseSettings):
    chat: ChatConfig = ChatConfig()
    OPENAI_API_KEY: str | None = None
    class Config:
        env_file = ".env" 


