import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage

from yaf_gpt.core import Settings
from yaf_gpt.scripts.langchain import ingest_documents
from yaf_gpt.scripts.langchain import build_chain


# load persisted store
embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="chroma_study",
    embedding_function=embed,
)

config = Settings()

my_retriever = ingest_documents(config=config)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
chain = build_chain(my_retriever, config=config)

def ask(question):
    result: AIMessage = chain.invoke(question)
    # result is usually an AIMessage; return its content
    return result.content if hasattr(result, "content") else str(result)

demo = gr.Interface(fn=ask, inputs=gr.Textbox(label="Question"), outputs="text")
demo.launch()
