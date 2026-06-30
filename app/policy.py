"""Deterministic decision layer. Takes the LLM's interpretation of the
conversation and turns it into a concrete action using plain Python control
flow — this is what keeps a non-deterministic LLM judgment from being able to
single-handedly blow the turn cap or skip a refusal."""
from dataclasses import dataclass
from typing import Literal

from app.config import MAX_CONVERSATION_MESSAGES, MAX_MESSAGES_BEFORE_FORCED_COMMIT
from app.retriever import SearchFilters
from app.schemas import ChatMessage
from app.slots import Interpretation

Action = Literal["refuse", "clarify", "answer"]


@dataclass
class PolicyDecision:
    action: Action
    end_of_conversation: bool
    refusal_category: str | None
    comparison_requested: bool
    compare_names: list[str]
    should_recommend: bool   # whether a shortlist should be attempted (independent of comparison_requested)
    force_commit: bool       # should_recommend is true only because the turn budget forced it
    retrieval_query: str
    filters: SearchFilters
    clarifying_question: str | None


def _build_retrieval_query(interpretation: Interpretation) -> str:
    c = interpretation.constraints
    parts = [c.role_or_skill, c.seniority, " ".join(c.keywords)]
    query = " ".join(p for p in parts if p)
    return query or "general assessment"


def _build_filters(interpretation: Interpretation) -> SearchFilters:
    c = interpretation.constraints
    return SearchFilters(
        test_types=set(c.test_types) if c.test_types else None,
        max_duration_minutes=c.max_duration_minutes,
        remote_only=bool(c.remote_required),
        language=c.language,
    )


def decide(messages: list[ChatMessage], interpretation: Interpretation) -> PolicyDecision:
    turns_so_far = len(messages)
    force_commit = turns_so_far >= MAX_MESSAGES_BEFORE_FORCED_COMMIT
    must_end_due_to_cap = turns_so_far >= MAX_CONVERSATION_MESSAGES - 1

    comparison_requested = interpretation.intent_compare and bool(interpretation.compare_names)

    # Whether to attempt a shortlist is decided independently of comparison_requested:
    # a context-free comparison question ("what's the difference between X and Y?")
    # should get a grounded text answer without the agent inventing an unrelated
    # shortlist just because it technically took the "answer" branch.
    should_recommend = (
        interpretation.ready_to_recommend
        or interpretation.prior_recommendations_given
        or interpretation.user_is_closing
        or force_commit
    )

    if not interpretation.in_scope:
        action: Action = "refuse"
        end_of_conversation = must_end_due_to_cap
    elif should_recommend or comparison_requested:
        action = "answer"
        end_of_conversation = interpretation.user_is_closing or must_end_due_to_cap
    else:
        action = "clarify"
        end_of_conversation = False

    return PolicyDecision(
        action=action,
        end_of_conversation=end_of_conversation,
        refusal_category=interpretation.refusal_category,
        comparison_requested=comparison_requested,
        compare_names=interpretation.compare_names,
        should_recommend=should_recommend,
        force_commit=force_commit and not (
            interpretation.ready_to_recommend
            or interpretation.prior_recommendations_given
            or interpretation.user_is_closing
        ),
        retrieval_query=_build_retrieval_query(interpretation),
        filters=_build_filters(interpretation),
        clarifying_question=interpretation.clarifying_question,
    )
