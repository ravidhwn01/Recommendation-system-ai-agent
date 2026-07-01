from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


class EmbeddingModel:

    def __init__(self):

        self.model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    def get_embedding_model(self):
        return self.model