"""LLM-driven SHL assessment agent.

Flow per /chat call (stateless):
  1. DECISION  - one LLM call classifies the next action from the full
     conversation and distills a search query (rule-based fallback if the LLM
     is unavailable).
  2. RECOMMEND/REFINE - retrieve catalog candidates, then a grounded LLM
     selection picks the final 1-10 (falls back to retrieval order). URLs are
     always taken from catalog metadata, never generated.
  3. COMPARE   - resolve two catalog assessments and produce a grounded LLM
     comparison (falls back to a deterministic catalog summary).
  4. CLARIFY / REFUSE - reply only, empty recommendations.
"""

from app.agent.comparator import ComparisonEngine
from app.agent.context import ContextManager
from app.agent.guardrails import GuardRails
from app.agent.intent import Intent, IntentClassifier
from app.agent.prompt_builder import (
    COMPARISON_SYSTEM,
    DECISION_SYSTEM,
    SELECTION_SYSTEM,
    comparison_user,
    decision_user,
    selection_user,
)
from app.core.logger import logger
from app.data.schemas import expand_test_type
from app.models.recommendation import Recommendation
from app.models.response import ChatResponse
from app.rag.retriever import Retriever
from app.services.llm_service import LLMService

CLARIFY_FALLBACK = (
    "Could you tell me a bit more about the role or skills you're hiring for? "
    "For example, 'a mid-level Java developer' or paste the job description, "
    "and I'll recommend suitable SHL assessments."
)
REFUSE_FALLBACK = (
    "Sorry, I can only help with SHL assessments. Tell me the role or skills "
    "you're hiring for and I'll recommend suitable assessments."
)

# Fallback-only signals (used when the LLM decision call is unavailable).
_SIGNALS = [
    "developer", "engineer", "manager", "analyst", "sales", "designer",
    "accountant", "nurse", "consultant", "scientist", "technician",
    "administrator", "executive", "clerk", "officer", "java", "python", "sql",
    "javascript", "excel", "coding", "graduate", "senior", "junior", "mid",
    "personality", "cognitive", "numerical", "verbal", "leadership",
]


class Agent:

    CANDIDATE_K = 20
    MAX_RECS = 10

    def __init__(self):
        self.llm = LLMService()
        self.retriever = Retriever()
        self.comparator = ComparisonEngine()
        self.context = ContextManager()
        self.guard = GuardRails()
        self.classifier = IntentClassifier()

    # -- public entry point -------------------------------------------------

    def handle(self, messages: list[dict]) -> ChatResponse:
        latest = self.context.get_latest_user_message(messages)
        if not latest:
            return ChatResponse(reply=CLARIFY_FALLBACK, recommendations=[],
                                end_of_conversation=False)

        conversation = self.context.get_conversation(messages)
        all_user_text = self.context.get_all_user_text(messages)
        assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")

        decision = self._decide(conversation, latest, all_user_text,
                                assistant_turns)
        action = decision.get("action", "clarify")

        if action == "refuse":
            # Fixed message: never let the model comply with an off-topic or
            # injection request, even partially.
            return ChatResponse(reply=REFUSE_FALLBACK, recommendations=[],
                                end_of_conversation=False)

        if action == "compare":
            return self._compare(latest, decision)

        if action in ("recommend", "refine"):
            need = decision.get("search_query") or all_user_text
            eoc = bool(decision.get("end_of_conversation", False))
            # On refine, also retrieve for the new constraint so the edit is
            # reflected in the candidate pool (not drowned out by prior terms).
            boost = latest if action == "refine" else None
            return self._recommend(need, decision, eoc, boost)

        # clarify (default)
        return ChatResponse(reply=decision.get("reply") or CLARIFY_FALLBACK,
                            recommendations=[], end_of_conversation=False)

    # -- decision -----------------------------------------------------------

    def _decide(self, conversation: str, latest: str, all_user_text: str,
                assistant_turns: int) -> dict:
        result = self.llm.generate_json(
            DECISION_SYSTEM, decision_user(conversation)
        )
        if not (isinstance(result, dict) and result.get("action") in {
            "recommend", "refine", "compare", "clarify", "refuse"
        }):
            logger.info("Decision LLM unavailable; using rule-based fallback.")
            result = self._fallback_decision(latest, all_user_text,
                                             assistant_turns)

        # Turn guard: never keep clarifying once we have context and have
        # already asked. Guarantees the agent commits within the turn cap.
        if (result.get("action") == "clarify"
                and assistant_turns >= 2
                and self._actionable(all_user_text)):
            logger.info("Turn guard: committing to a shortlist.")
            result = {"action": "recommend", "search_query": all_user_text}

        return result

    def _fallback_decision(self, latest: str, all_user_text: str,
                           assistant_turns: int) -> dict:
        if self.guard.is_injection(latest) or self.guard.is_off_topic(latest):
            return {"action": "refuse", "reply": REFUSE_FALLBACK}

        intent = self.classifier.classify(latest).intent
        if intent == Intent.COMPARE:
            return {"action": "compare", "compare_terms": []}
        if intent == Intent.REFINE and self._actionable(all_user_text):
            return {"action": "refine", "search_query": all_user_text}
        if not self._actionable(all_user_text):
            if intent == Intent.GREETING:
                return {"action": "clarify",
                        "reply": "Hello! Tell me the role you're hiring for "
                                 "and I'll recommend suitable SHL assessments."}
            return {"action": "clarify", "reply": CLARIFY_FALLBACK}

        # Actionable: ask one clarifying question first, then recommend.
        if assistant_turns == 0:
            return {"action": "clarify",
                    "reply": "Got it. What seniority level are you hiring for "
                             "(e.g. entry, mid, or senior), and are there "
                             "specific skills or competencies to prioritize?"}
        return {"action": "recommend", "search_query": all_user_text}

    @staticmethod
    def _actionable(text: str) -> bool:
        lowered = text.lower()
        if len(lowered.split()) >= 12:
            return True
        return any(sig in lowered for sig in _SIGNALS)

    # -- recommend / refine -------------------------------------------------

    def _recommend(self, need: str, decision: dict, eoc: bool = False,
                   boost: str | None = None) -> ChatResponse:
        docs = self.retriever.search(need, k=self.CANDIDATE_K)

        if boost:
            docs = self._merge(self.retriever.search(boost, k=10), docs)

        docs = [d for d in docs if self._valid(d.metadata)]

        if not docs:
            return ChatResponse(reply=CLARIFY_FALLBACK, recommendations=[],
                                end_of_conversation=False)

        chosen, reply = self._select(need, docs)
        if not chosen:
            chosen = docs[: self.MAX_RECS]

        recs, seen = [], set()
        for doc in chosen:
            name = doc.metadata["name"]
            if name in seen:
                continue
            seen.add(name)
            recs.append(Recommendation(
                name=name,
                url=doc.metadata["url"],
                test_type=(doc.metadata.get("test_type") or "").strip(),
            ))
            if len(recs) >= self.MAX_RECS:
                break

        if not reply:
            reply = decision.get("reply") or (
                f"Here are {len(recs)} SHL assessments that fit your needs."
            )

        return ChatResponse(reply=reply, recommendations=recs,
                            end_of_conversation=eoc)

    def _select(self, need: str, docs: list):
        """LLM grounded selection -> (chosen_docs, reply). Falls back to []."""
        block = "\n".join(
            f"[{i + 1}] {self._candidate_line(d.metadata)}"
            for i, d in enumerate(docs)
        )
        result = self.llm.generate_json(
            SELECTION_SYSTEM, selection_user(need, block)
        )
        if not isinstance(result, dict):
            return [], None

        indices = result.get("indices") or []
        chosen = []
        for idx in indices:
            if isinstance(idx, int) and 1 <= idx <= len(docs):
                chosen.append(docs[idx - 1])

        return chosen, result.get("reply")

    @staticmethod
    def _candidate_line(meta: dict) -> str:
        bits = [meta.get("name", "")]
        if meta.get("test_type"):
            readable = expand_test_type(meta["test_type"])
            bits.append(f"type: {readable or meta['test_type']}")
        if meta.get("duration"):
            bits.append(f"{meta['duration']} min")
        if meta.get("job_levels"):
            bits.append(f"levels: {meta['job_levels']}")
        return " | ".join(bits)

    # -- compare ------------------------------------------------------------

    def _compare(self, latest: str, decision: dict) -> ChatResponse:
        terms = decision.get("compare_terms") or []
        query = " and ".join(terms) if terms else latest

        targets = self.comparator.resolve_targets(query)
        if len(targets) < 2:
            targets = self.comparator.resolve_targets(latest)

        if len(targets) < 2:
            return ChatResponse(
                reply="I couldn't find two matching SHL assessments to "
                      "compare. Which assessments would you like compared?",
                recommendations=[], end_of_conversation=False)

        reply = self.llm.generate_text(
            COMPARISON_SYSTEM,
            comparison_user(
                self.comparator.format_target(targets[0]),
                self.comparator.format_target(targets[1]),
                latest,
            ),
        )
        if not reply:
            reply = self.comparator.compare_reply(query)

        return ChatResponse(reply=reply, recommendations=[],
                            end_of_conversation=False)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _merge(primary: list, secondary: list) -> list:
        """Interleave two candidate lists, deduped by name, primary first."""
        merged, seen = [], set()
        for doc in [d for pair in zip(primary, secondary) for d in pair] \
                + primary + secondary:
            name = doc.metadata.get("name")
            if name and name not in seen:
                seen.add(name)
                merged.append(doc)
        return merged

    @staticmethod
    def _valid(meta: dict) -> bool:
        url = (meta.get("url") or "").strip()
        return bool(meta.get("name")) and url.startswith("http") \
            and "shl.com" in url
