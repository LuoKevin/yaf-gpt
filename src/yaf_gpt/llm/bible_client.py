from openai import OpenAI

from yaf_gpt.core.config import Settings


def bible_client(config: Settings | None = None) -> OpenAI:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=config.HF_TOKEN if config else None,
    )

    return client

if __name__ == "__main__":
    config = Settings()
    client = bible_client(config=config)
    completion = client.chat.completions.create(
        model="sleepdeprived3/Reformed-Christian-Bible-Expert-12B:featherless-ai",
        messages=[
            {
                "role": "system",
                "content": """
                    Given a bible passage, provide the following 
                    - The raw passage text itself
                    - Provide brief context and background of the setting as the passage takes place (do not summarize the passage)
                    - Provide discussion questions to ask so the user can grasp and comprehend what is happening, and the lessons that can be learned from them
                    - Ask the user how they can take those lessons and themes and apply them in their own life.
                """
            },
            {
                "role": "user",
                "content": "What is the tale of the prodigal son?"
            }
        ],
    )

    print(completion.choices[0].message.content)