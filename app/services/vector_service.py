from langchain_chroma import Chroma

from app.core.config import settings
from app.rag.embedding import EmbeddingModel


class VectorService:

    def __init__(self):

        embedding = EmbeddingModel().get_embedding_model()

        self.db = Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=embedding,
        )

    def get_vector_store(self):
        return self.db

    def similarity_search(self, query: str, k: int = 10):
        return self.db.similarity_search(query, k=k)

    def get_all_metadatas(self) -> list[dict]:
        result = self.db.get()
        return result.get("metadatas") or []

    def add_documents(self, documents):
        self.db.add_documents(documents)