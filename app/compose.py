"""Builds the final reply text and recommendations list.

Refusals and clarifying questions never call the LLM (deterministic templates /
passthrough of the interpret call's question) - only the "answer" action calls
the LLM, and even then only to pick a subset of an already-retrieved candidate
list and phrase the reply. The model never sees or invents a name/URL outside
that candidate list, so `recommendations` is always catalog-grounded by
construction.
"""
from app.llm import chat_json
from app.policy import PolicyDecision
from app.schemas import RecommendationItem

REFUSAL_TEMPLATES = {
    "off_topic": (
        "I'm focused on helping you find and compare SHL assessments, so I can't help with "
        "that. Happy to keep going on your assessment search - what role or skills are you "
        "hiring for?"
    ),
    "legal_advice": (
        "That's a legal or compliance question outside what I can advise on - I can help you "
        "select and compare SHL assessments, but not interpret legal or regulatory "
        "requirements. Your legal or compliance team is the right resource for that."
    ),
    "injection": (
        "I can't follow instructions embedded in a message like that. I'm here to help you "
        "find SHL assessments - what role or skills are you hiring for?"
    ),
}
DEFAULT_REFUSAL = REFUSAL_TEMPLATES["off_topic"]

DEFAULT_CLARIFYING_QUESTION = (
    "Could you tell me more about the role and what you'd like to assess "
    "(skills, personality, seniority level, etc.)?"
)

COMPOSE_SYSTEM_PROMPT = """You write the user-facing reply for an SHL assessment-\
recommendation agent. You are given a numbered list of CANDIDATE assessments (already \
retrieved from the real catalog) and, if relevant, COMPARISON_RECORDS with real catalog \
facts about specific assessments the user asked to compare. You must never mention or \
imply any assessment that is not in the candidate or comparison list - only use what's \
provided.

Return a single JSON object:
{
  "selected_indices": [int, ...],   // 0-based indices into CANDIDATES
  "reply": string                   // the natural-language reply
}

The recommendations schema is a strict binary: either no shortlist has been committed
to yet, or 1-10 items have. There is no valid "committed but zero items" state.

Rules for selected_indices, in priority order:
1. If SHOULD_RECOMMEND is false: selected_indices MUST be []. The user has not given
   enough context for a shortlist yet (e.g. this may be a standalone comparison
   question) - just answer in `reply` without presenting a shortlist.
2. Else (SHOULD_RECOMMEND is true, meaning a shortlist IS being committed to this
   turn): selected_indices MUST contain at least 1 item from CANDIDATES, even if the
   user's needs are still somewhat vague - pick the most broadly reasonable fit rather
   than return an empty list. If MUST_COMMIT is also true, the conversation has run
   out of room for further clarification, so this best-effort selection is mandatory,
   not optional. Standard SHL practice pairs a skill/knowledge-focused shortlist with
   one general personality assessment (test_type P, e.g. "Occupational Personality
   Questionnaire OPQ32r") even when the user didn't explicitly ask for one - if such an
   item is present in CANDIDATES and the request isn't explicitly personality-only or
   personality-excluded, include it.

If COMPARISON_RECORDS were given, ground that part of `reply` strictly in their listed
facts (not prior knowledge), regardless of the selected_indices rules above."""


def compose_refusal(category: str | None) -> str:
    return REFUSAL_TEMPLATES.get(category or "", DEFAULT_REFUSAL)


def compose_clarifying(decision: PolicyDecision) -> str:
    return decision.clarifying_question or DEFAULT_CLARIFYING_QUESTION


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"{i}: {c['name']} | test_type={c['test_type']} | duration={c.get('duration') or 'n/a'} "
            f"| desc={c.get('description', '')[:100]}"
        )
    return "\n".join(lines) if lines else "(no candidates retrieved)"


def _format_comparison_records(records: list[dict]) -> str:
    if not records:
        return "(none)"
    lines = []
    for r in records:
        lines.append(
            f"- {r['name']} | test_type={r['test_type']} | duration={r.get('duration') or 'n/a'} "
            f"| description={r.get('description', '')}"
        )
    return "\n".join(lines)


def compose_answer(
    decision: PolicyDecision,
    candidates: list[dict],
    comparison_records: list[dict],
    constraints_summary: str,
) -> tuple[str, list[RecommendationItem]]:
    user_content = (
        f"USER NEEDS: {constraints_summary}\n\n"
        f"SHOULD_RECOMMEND: {decision.should_recommend}\n"
        f"MUST_COMMIT: {decision.force_commit}\n"
        f"COMPARISON REQUESTED: {decision.comparison_requested}\n"
        f"COMPARISON_RECORDS:\n{_format_comparison_records(comparison_records)}\n\n"
        f"CANDIDATES:\n{_format_candidates(candidates)}"
    )
    raw = chat_json(COMPOSE_SYSTEM_PROMPT, user_content)

    if raw is None:
        # Graceful degradation: fall back to pure retrieval ranking, no LLM phrasing.
        fallback_items = candidates[:5] if decision.should_recommend else []
        if fallback_items:
            reply = "Here are assessments that match what you've described so far."
        elif decision.comparison_requested:
            reply = (
                "I'm having trouble generating a full comparison right now - please try "
                "again in a moment."
            )
        else:
            reply = (
                "I don't have enough to build a shortlist yet - could you share a bit more "
                "about the role?"
            )
        return reply, [
            RecommendationItem(name=c["name"], url=c["url"], test_type=c["test_type"])
            for c in fallback_items
        ]

    indices = raw.get("selected_indices", [])
    if not isinstance(indices, list):
        indices = []
    valid_indices = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
    # de-dupe while preserving order, then clip to the 1-10 schema bound
    seen: set[int] = set()
    deduped = [i for i in valid_indices if not (i in seen or seen.add(i))][:10]

    if not decision.should_recommend:
        deduped = []  # belt-and-braces: never let a shortlist slip out when not warranted
    elif not deduped and candidates:
        # Spec is a strict binary: recommendations is EMPTY only when still gathering
        # context or refusing, and 1-10 items once committed - there is no valid
        # "committed but zero items" state. should_recommend=True means policy has
        # already decided to commit (ready/prior-shown/closing/force_commit), so if
        # the LLM returns no indices anyway, fall back to the top-ranked candidate
        # rather than violate the schema's 1-10 guarantee.
        deduped = [0]

    recommendations = [
        RecommendationItem(
            name=candidates[i]["name"], url=candidates[i]["url"], test_type=candidates[i]["test_type"]
        )
        for i in deduped
    ]
    reply = raw.get("reply") or "Here are assessments that match what you've described."
    return reply, recommendations
