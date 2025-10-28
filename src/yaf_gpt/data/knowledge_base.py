import json
from pathlib import Path
from yaf_gpt.data.study_chunk import StudyChunk
from sklearn.feature_extraction.text import TfidfVectorizer

class KnowledgeBase:
    """Manages the knowledge base for yaf_gpt."""

    def __init__(self, data_path: Path | str) -> None:
        self._vectorizer = TfidfVectorizer(stop_words='english')
        self.data_path = Path(data_path)
        self.chunks: list[StudyChunk] = self._load_knowledge_base()
        self._vectors = self._vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    def _load_knowledge_base(self) -> list[StudyChunk]:
        """Load knowledge base chunks from the specified data path."""
        chunks: list[StudyChunk] = []
        with open(self.data_path, "r", encoding="utf-8") as fh:
            for line in fh:
                chunks.append(StudyChunk(**json.loads(line)))
        return chunks

    def get_chunks(self) -> list[StudyChunk]:
        """Return the loaded knowledge base chunks."""
        return self.chunks

    def search(self, query: str, top_k: int = 3) -> list[tuple[StudyChunk, float]]:
        """Search the knowledge base for the most relevant chunks to the query."""
        top_k = min(top_k, len(self.chunks))
        query_vector = self._vectorizer.transform([query])
        similarities = (self._vectors @ query_vector.T).toarray().flatten()
        if not similarities.any():
            return []
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(self.chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0]
