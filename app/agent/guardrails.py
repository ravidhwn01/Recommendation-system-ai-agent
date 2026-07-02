import re


class GuardRails:
    """Keeps the agent in scope: SHL assessments only.

    Refuses prompt-injection attempts, clearly off-topic questions, and
    general hiring/legal advice that falls outside the assessment catalog.
    """

    INJECTION_PATTERNS = [
        r"ignore\s+(all|any|the\s+)?(previous|prior|above)",
        r"disregard\s+(all|any|the\s+)?(previous|prior|above|instructions)",
        r"system\s+prompt",
        r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)",
        r"(show|print|repeat)\s+(me\s+)?(your|the)\s+(prompt|instructions)",
        r"you\s+are\s+now\b",
        r"pretend\s+(to\s+be|you\s+are)",
        r"act\s+as\s+(a|an|if)\b",
        r"jailbreak",
        r"developer\s+mode",
        r"forget\s+(all|everything|your\s+instructions)",
    ]

    OFF_TOPIC_KEYWORDS = [
        "weather", "capital of", "recipe", "cook", "movie", "song",
        "lyrics", "poem", "cricket", "football", "ipl", "soccer", "nba",
        "bitcoin", "crypto", "stock market", "president", "election",
        "politics", "joke", "horoscope", "who won", "translate this",
    ]

    ADVICE_PATTERNS = [
        r"\blegal\b",
        r"\blaw(s|suit|suits)?\b",
        r"employment\s+law",
        r"labou?r\s+law",
        r"discriminat",
        r"how\s+(do|should|can)\s+i\s+(hire|fire|interview)",
        r"hiring\s+(advice|tips|law|laws)",
        r"(should|can)\s+i\s+(fire|reject|reduce\s+salary)",
        r"salary\s+(negotiat|expectation|range\s+for)",
    ]

    def is_injection(self, query: str) -> bool:
        text = query.lower()
        return any(re.search(p, text) for p in self.INJECTION_PATTERNS)

    def is_off_topic(self, query: str) -> bool:
        text = query.lower()
        if any(word in text for word in self.OFF_TOPIC_KEYWORDS):
            return True
        if any(re.search(p, text) for p in self.ADVICE_PATTERNS):
            return True
        return False

    def is_allowed(self, query: str) -> bool:
        return not (self.is_injection(query) or self.is_off_topic(query))
