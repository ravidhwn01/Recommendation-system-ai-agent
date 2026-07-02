OFF_TOPIC_KEYWORDS = [
    "weather",
    "ipl",
    "cricket",
    "football",
    "movie",
    "recipe",
    "politics",
    "salary",
    "bitcoin",
]


class GuardRails:

    def is_allowed(self, query: str) -> bool:

        query = query.lower()

        for word in OFF_TOPIC_KEYWORDS:
            if word in query:
                return False

        return True