from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from yaf_gpt.config import Settings

def _build_context(docs: list[Document]):
    return "\n\n".join(f"{doc.metadata.get('source', 'unknown')} \n {doc.page_content[:500]}" for doc in docs)


def build_chain(retriever: VectorStoreRetriever, config: Settings | None = None, openai: ChatOpenAI | None = None) -> Runnable:
    """Build a retrieval-augmented generation chain."""
    if openai is None:
        model_name = config.chat.model if config else "gpt-4o-mini"
        openai = ChatOpenAI(
            model_name=model_name,
            temperature=config.chat.temperature if config else 0.0,
        )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a church minister leading a a fellowship of young adults. 
                You seek to help young adults becoming rooted in the gospel to reflect the glory of God.
                Your mission is to build up God-loving, church-serving disciples of Christ.
            """),
            ("human", "Context: \n {context} \n\nQuestion: {question}")
            ]
    )

    chain = (
        {"question": RunnablePassthrough(),
         "context": retriever |  RunnableLambda(_build_context)}
         | prompt_template
         | openai
    ).with_config({"run_name": "yaf_gpt_rag_chain"})

    return chain