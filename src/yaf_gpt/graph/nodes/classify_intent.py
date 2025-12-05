from jinja2 import Environment, FileSystemLoader

from src.yaf_gpt.services.factories import get_openai_client

def classify_intent(text):
    client, error = get_openai_client()
    if error or not client:
        raise ValueError(f"OpenAI client initialization failed: {error}")

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("classify_intent.jinja")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": template.render()
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    return response.choices[0].message.content.strip()
