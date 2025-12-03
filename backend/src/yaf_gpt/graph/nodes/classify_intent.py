

from backend.src.yaf_gpt.services.factories import get_openai_client


def classify_intent(text):

    client, error = get_openai_client()
    if error or not client:
        raise ValueError(f"OpenAI client initialization failed: {error}")
    
    