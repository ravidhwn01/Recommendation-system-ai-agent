from app.rag.retriever import Retriever


class RecommendationEngine:

    def __init__(self):

        self.retriever = Retriever()

    def recommend(
        self,
        query: str,
        k: int = 10,
    ):

        return self.retriever.search(query)[:k]