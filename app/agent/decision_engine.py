from app.agent.intent import Intent


class DecisionEngine:

    def decide(self, intent: Intent) -> str:

        if intent == Intent.RECOMMEND:
            return "recommend"

        if intent == Intent.COMPARE:
            return "compare"

        if intent == Intent.REFINE:
            return "refine"

        if intent == Intent.GREETING:
            return "greet"

        return "reject"