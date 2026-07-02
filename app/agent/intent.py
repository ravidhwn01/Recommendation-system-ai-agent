import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    GREETING = "GREETING"
    RECOMMEND = "RECOMMEND"
    COMPARE = "COMPARE"
    REFINE = "REFINE"
    OFF_TOPIC = "OFF_TOPIC"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float


def _has(text: str, words: list[str]) -> bool:
    """Whole-word match so 'hi' does not match 'hiring'."""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


class IntentClassifier:

    COMPARE_WORDS = ["compare", "comparison", "difference", "differences",
                     "vs", "versus", "better"]
    REFINE_WORDS = ["actually", "instead", "also", "additionally",
                    "add", "remove", "change", "as well", "on second thought"]
    GREETING_WORDS = ["hi", "hello", "hey", "yo", "hiya",
                      "good morning", "good afternoon", "good evening"]

    def classify(self, query: str) -> IntentResult:
        text = query.strip().lower()

        if _has(text, self.COMPARE_WORDS):
            return IntentResult(Intent.COMPARE, 0.97)

        if _has(text, self.REFINE_WORDS):
            return IntentResult(Intent.REFINE, 0.9)

        # Greeting only when the message is essentially just a greeting
        # (short and starts with a greeting token).
        if _has(text, self.GREETING_WORDS) and len(text.split()) <= 4:
            return IntentResult(Intent.GREETING, 0.95)

        return IntentResult(Intent.RECOMMEND, 0.6)
