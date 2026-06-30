"""Entry point for the agent core: guard -> interpret -> policy -> retrieve ->
compose. main.py's /chat handler calls handle_chat() directly."""
from dataclasses import replace

from app.compose import compose_answer, compose_clarifying, compose_refusal
from app.guard import is_injection_attempt
from app.policy import PolicyDecision, decide
from app.retriever import CatalogIndex, get_index
from app.schemas import ChatMessage, ChatResponse
from app.slots import Constraints, interpret_conversation

CANDIDATE_POOL_SIZE = 20


def _retrieve_candidates(index: CatalogIndex, decision: PolicyDecision) -> list[dict]:
    """A single BM25 query biased toward role/skill terms (e.g. "Java developer")
    ranks the matching test_type's items far above any other requested type's
    items, even when a filter technically allows both - a generic personality
    test scores ~0 against a Java-heavy query and never reaches the top_k. When
    more than one test_type is requested, retrieve per type and merge so every
    requested category gets fair representation in the candidate pool."""
    types = decision.filters.test_types
    if not types or len(types) <= 1:
        return index.search(decision.retrieval_query, filters=decision.filters, top_k=CANDIDATE_POOL_SIZE)

    per_type_k = max(CANDIDATE_POOL_SIZE // len(types), 5)
    seen_ids: set[str] = set()
    merged: list[dict] = []
    for test_type in sorted(types):
        sub_filters = replace(decision.filters, test_types={test_type})
        for item in index.search(decision.retrieval_query, filters=sub_filters, top_k=per_type_k):
            if item["entity_id"] not in seen_ids:
                seen_ids.add(item["entity_id"])
                merged.append(item)
    return merged


def _last_user_message(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _summarize_constraints(c: Constraints) -> str:
    parts = []
    if c.role_or_skill:
        parts.append(f"role/skill: {c.role_or_skill}")
    if c.seniority:
        parts.append(f"seniority: {c.seniority}")
    if c.test_types:
        parts.append(f"requested test types: {', '.join(c.test_types)}")
    if c.max_duration_minutes:
        parts.append(f"max duration: {c.max_duration_minutes} minutes")
    if c.remote_required:
        parts.append("remote required")
    if c.language:
        parts.append(f"language: {c.language}")
    if c.keywords:
        parts.append(f"other keywords: {', '.join(c.keywords)}")
    return "; ".join(parts) if parts else "no specific constraints stated yet"


def handle_chat(messages: list[ChatMessage]) -> ChatResponse:
    if not messages:
        return ChatResponse(
            reply=compose_refusal("off_topic"), recommendations=[], end_of_conversation=False
        )

    if is_injection_attempt(_last_user_message(messages)):
        return ChatResponse(
            reply=compose_refusal("injection"), recommendations=[], end_of_conversation=False
        )

    interpretation = interpret_conversation(messages)
    decision = decide(messages, interpretation)

    if decision.action == "refuse":
        return ChatResponse(
            reply=compose_refusal(decision.refusal_category),
            recommendations=[],
            end_of_conversation=decision.end_of_conversation,
        )

    if decision.action == "clarify":
        return ChatResponse(
            reply=compose_clarifying(decision), recommendations=[], end_of_conversation=False
        )

    # action == "answer"
    index = get_index()
    candidates = []
    if decision.should_recommend:
        candidates = _retrieve_candidates(index, decision)

    comparison_records = []
    if decision.comparison_requested:
        for name in decision.compare_names:
            comparison_records.extend(index.find_by_name(name, top_k=1))

    reply, recommendations = compose_answer(
        decision,
        candidates,
        comparison_records,
        _summarize_constraints(interpretation.constraints),
    )

    return ChatResponse(
        reply=reply, recommendations=recommendations, end_of_conversation=decision.end_of_conversation
    )
