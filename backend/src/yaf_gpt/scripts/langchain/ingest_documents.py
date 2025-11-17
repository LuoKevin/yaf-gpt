from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
import random
import re

from yaf_gpt.core.config import Settings

LINE_MARKER = "###"

def _inspect_docs(docs: list[Document]):
    #choose a few random documents to inspect
    random_docs = random.sample(docs, min(5, len(docs)))
    
    for doc in random_docs:
        print(f"--- Document (metadata: {doc.metadata}) ---")
        print(doc.page_content[:500])
        print()


def process_doc(doc: Document) -> Document:
    tagged_text = tag_sections(doc.page_content)
    source = doc.metadata.get("source", "unknown.docx")
    file_name = Path(source).name.replace(".docx", "")
    doc.metadata["passage_reference"] = file_name
    doc.metadata["book"] = file_name.split(" ")[0]
    return Document(page_content=tagged_text, metadata=doc.metadata)

def is_section_marker(line: str) -> bool:
    stripped_line = line.strip().upper()
    regex = r"(QUESTIONS|CONTEXT|APPLICATION|PASSAGE)"
    return re.match(regex, stripped_line) and len(stripped_line) < 30

def tag_sections(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if is_section_marker(line):
            lines.append(f"{LINE_MARKER} {line.strip()}")
        else:
            lines.append(line)
    return "\n".join(lines)


def ingest_documents(config: Settings | None = None) -> VectorStoreRetriever:
    DATA_DIR = Path("raw_data") / "Bible_Study_Docs" / "Luke"

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
            LINE_MARKER,     # prefer splitting on section markers
            "\n\n",          # prefer splitting on blank lines
            "\n",            # then single newline
            " ",             # fall back to whitespace
            ""     
        ]
    )

    docs = [process_doc(doc) for doc in loader.load()]

    chunks = text_splitter.split_documents(docs)

    embeddings = (
        OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small",
        )
        if config and config.OPENAI_API_KEY
        else HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="data/chroma_study")
    vector_store.persist()

    retriever: VectorStoreRetriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever


# if __name__ == "__main__":
#     ingest_documents()
