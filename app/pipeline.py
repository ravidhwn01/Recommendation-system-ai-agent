"""Entry point for the agent core: guard -> interpret -> policy -> retrieve ->
compose. main.py's /chat handler calls handle_chat() directly."""
from app.compose import compose_answer, compose_clarifying, compose_refusal
from app.guard import is_injection_attempt
from app.policy import decide
from app.retriever import get_index
from app.schemas import ChatMessage, ChatResponse
from app.slots import Constraints, interpret_conversation

CANDIDATE_POOL_SIZE = 20


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
        candidates = index.search(decision.retrieval_query, filters=decision.filters, top_k=CANDIDATE_POOL_SIZE)

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
