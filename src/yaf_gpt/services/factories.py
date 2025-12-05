

from functools import lru_cache
from typing import Optional, Tuple

from langchain_openai import OpenAI

from src.yaf_gpt.core.config import Settings

@lru_cache
def get_settings() -> Settings:
    return Settings()

def get_openai_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    try:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            return None, "openai_api_key_missing"
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return openai_client, None
    except Exception as e:
        return None, str(e) 