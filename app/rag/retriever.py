from app.services.vector_service import VectorService


class Retriever:

    def __init__(self):
        self.vector_store = VectorService().get_vector_store()

    def search(
        self,
        query: str,
        k: int = 10,
    ):
        return self.vector_store.similarity_search(
            query,
            k=k,
        )