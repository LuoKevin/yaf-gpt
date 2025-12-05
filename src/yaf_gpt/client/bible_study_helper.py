
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

from yaf_gpt.llm.bible_client import openai_client


class BibleStudyHelper:

    client: OpenAI

    JINJA_TEMPLATE_DIR = "src/yaf_gpt/templates"
    env = Environment(loader=FileSystemLoader(JINJA_TEMPLATE_DIR))

    def __init__(self, client):
        self.client = client

    def icebreaker(self, passage) -> str:
        template = self.env.get_template("icebreaker.jinja")
        prompt = template.render(passage_reference=passage)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content

    def questions(self, passage) -> str:
        template = self.env.get_template("passage_questions.jinja")
        prompt = template.render(passage_reference=passage)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content

    def passage_text(self, passage):
        template = self.env.get_template("bible_passage.jinja")
        prompt = template.render(passage_reference=passage)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content
    


    def life_application(self, passage) -> str:
        template = self.env.get_template("life_application.jinja")
        prompt = template.render(passage_reference=passage)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content

    def image_study(self, passage):
        result = client.images.generate(
            model="gpt-image-1",
            prompt=f"An inspiring and motivational scene from the bible passage: {passage}",
            n=1,
            size="512x512"
        )
        image_base64 = result.data[0].b64_json
        return image_base64

    def study(self, reference):
        env = Environment(loader=FileSystemLoader("src/yaf_gpt/templates"))
        template = env.get_template("study_notes.jinja")
        system_prompt = template.render()
        print(system_prompt)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
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

    # print(helper.icebreaker("Luke 10:25"))
    # print(helper.study("Luke 10:25"))
    print(helper.life_application("Luke 15:11-32"))


    # study_response = helper.study("Luke 10:25")

    # print(study_response.choices[0].message.content)