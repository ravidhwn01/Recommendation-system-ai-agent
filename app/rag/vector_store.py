from langchain_chroma import Chroma

from app.core.config import settings

from app.rag.embedding import EmbeddingModel


class VectorStore:

    def __init__(self):

        embedding_model = EmbeddingModel()

        self.db = Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=embedding_model.get_embedding_model()
        )

    def add_documents(self, documents):

        self.db.add_documents(documents)

    def get_vector_store(self):

        return self.db