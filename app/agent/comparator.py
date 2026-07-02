from app.rag.retriever import Retriever


class ComparisonEngine:

    def __init__(self):
        self.retriever = Retriever()

    def compare(
        self,
        assessment1: str,
        assessment2: str,
    ):

        docs1 = self.retriever.search(assessment1)

        docs2 = self.retriever.search(assessment2)

        if not docs1 or not docs2:
            return None

        return {
            "assessment_1": docs1[0],
            "assessment_2": docs2[0],
        }