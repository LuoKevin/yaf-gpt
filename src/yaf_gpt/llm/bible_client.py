import os
from openai import OpenAI

from yaf_gpt.core.config import Settings


def bible_client(config: Settings | None = None) -> OpenAI:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=config.HF_TOKEN if config else None,
        model="sleepdeprived3/Reformed-Christian-Bible-Expert-12B:featherless-ai",
    )

    return client
