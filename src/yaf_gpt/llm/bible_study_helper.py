
from openai import OpenAI
from jinja2 import Template
from langchain_core.prompts import ChatPromptTemplate


class BibleStudyHelper:

    client: OpenAI

    def __init__(self, client):
        self.client = client

    def _get_passage(self, reference):
        response = self.client.chat.completions.create(
            model="sleepdeprived3/Reformed-Christian-Bible-Expert-12B:featherless-ai",
            messages=[
                {
                    "role": "user",
                    "content": f"Provide the full text of the Bible passage (CSB edition) for the reference: {reference}"
                }
            ]
        )
        return response.choices[0].message.content

    def study(self, passage):
        response = self.client.chat.completions.create(
            model="sleepdeprived3/Reformed-Christian-Bible-Expert-12B:featherless-ai",
            messages=[
                {
                    "role": "system",
                    "content": ChatPromptTemplate.from_file("src/yaf_gpt/templates/task.jinja")
                },
                {
                    "role": "user",
                    "content": passage
                }
            ]
        )
        return response
    
if __name__ == "__main__":
    from yaf_gpt.core.config import Settings
    from yaf_gpt.llm.bible_client import bible_client

    config = Settings()
    client = bible_client(config=config)
    helper = BibleStudyHelper(client=client)

    passage_text = helper._get_passage("Luke 15:11-32")
    study_response = helper.study(passage_text)

    print(study_response.choices[0].message.content)