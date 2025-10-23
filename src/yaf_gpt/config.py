from typing import List
from pydantic.v1 import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    class Config:
        env_file = ".env"
