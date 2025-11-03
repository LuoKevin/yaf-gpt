from pydantic import BaseModel
from pydantic.v1 import BaseSettings

class ChatConfig(BaseModel):
    model: str = "gpt-4o-mini"
    system_prompt: str | None = None
    temperature: float  = 0.0
    max_tokens: int = 30
    knowledge_top_k: int = 3
    knowledge_enabled: bool = False
    max_context_chars: int = 4000

class Settings(BaseSettings):
    chat: ChatConfig = ChatConfig()
    OPENAI_API_KEY: str | None = None
    class Config:
        env_file = ".env" 


