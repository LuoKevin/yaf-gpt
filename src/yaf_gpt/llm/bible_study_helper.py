
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

from yaf_gpt.llm.bible_client import openai_client


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

    def depr_study(self, passage):
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

    def study(self, reference):
        env = Environment(loader=FileSystemLoader("src/yaf_gpt/templates"))
        template = env.get_template("study_notes.jinja")
        system_prompt = template.render()
        print(system_prompt)

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content":system_prompt
                },
                {
                    "role": "user",
                    "content": reference
                }
            ]
        )

        return response
    
if __name__ == "__main__":
    from yaf_gpt.core.config import Settings
    from yaf_gpt.llm.bible_client import bible_client

    config = Settings()
    client = openai_client(config=config)
    helper = BibleStudyHelper(client=client)

    study_response = helper.study("Luke 10:25")

    print(study_response.choices[0].message.content)