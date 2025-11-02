from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from yaf_gpt.config import Settings

def ingest_documents(config: Settings | None = None) -> Chroma:
    DATA_DIR = Path("raw_data") / "Bible_Study_Docs"

    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",          # prefer splitting on blank lines
            "\n",            # then single newline
            " ",             # fall back to whitespace
            ""     
        ]
    )

    chunks = loader.load_and_split(text_splitter)

    print(f"metadata: {chunks[0].metadata}")
    embeddings = (
        OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small",
        )
        if config
        else HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


if __name__ == "__main__":
    ingest_documents()
