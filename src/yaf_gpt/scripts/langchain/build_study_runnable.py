from jinja2 import Environment, FileSystemLoader
from langchain_core.runnables import Runnable, RunnableLambda

env = Environment(loader=FileSystemLoader("src/yaf_gpt/templates"))
template = env.get_template("study_notes.jinja")

def render_prompt(inputs: dict) -> dict:

    prompt_text = template.render(
        passage_reference=inputs.get("passage_reference", "Unknown Reference"),
        notes=inputs.get("notes", []),   # list of dicts/objects with title/content/metadata
        tone=inputs.get("tone", "warm and pastoral"),
    )
    return {**inputs,"prompt": prompt_text}

def build_study_runnable(runnable: Runnable) -> Runnable:
    return RunnableLambda(render_prompt) | runnable
        