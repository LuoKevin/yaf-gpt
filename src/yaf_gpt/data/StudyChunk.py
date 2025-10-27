
from pydantic import BaseModel


class StudyChunk(BaseModel):
    book: str
    doc_id: str
    section: str
    chunk_index: int
    text: str
    source_path: str