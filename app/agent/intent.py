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


class IntentClassifier:

    def classify(self, query: str) -> IntentResult:

        text = query.lower()

        if any(
            word in text
            for word in [
                "compare",
                "difference",
                "vs",
            ]
        ):

            return IntentResult(
                Intent.COMPARE,
                0.98,
            )

        if any(
            word in text
            for word in [
                "actually",
                "instead",
                "change",
                "also",
            ]
        ):

            return IntentResult(
                Intent.REFINE,
                0.95,
            )

        if any(
            word in text
            for word in [
                "assessment",
                "hire",
                "developer",
                "recommend",
                "test",
            ]
        ):

            return IntentResult(
                Intent.RECOMMEND,
                0.96,
            )

        if any(
            word in text
            for word in [
                "hello",
                "hi",
                "hey",
            ]
        ):

            return IntentResult(
                Intent.GREETING,
                0.99,
            )

        return IntentResult(
            Intent.OFF_TOPIC,
            0.80,
        )