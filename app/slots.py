"""The single "interpret" LLM call: reads the full conversation history (the
service is stateless, so this must be re-derived from scratch every request)
and produces a structured read of scope, intent, accumulated constraints, and
turn-state. Policy then makes the deterministic action decision on top of this.
"""
from dataclasses import dataclass, field

from app.llm import chat_json
from app.schemas import ChatMessage

SYSTEM_PROMPT = """You are the understanding layer of an SHL assessment-recommendation \
agent. You do not write the user-facing reply — you only extract structured state from \
the conversation so far. Always respond with a single JSON object, no other text.

SCOPE: this agent only helps select and compare SHL pre-built individual assessment \
products (cognitive/ability tests, personality questionnaires, knowledge & skills tests, \
biodata/situational judgment tests, simulations, development reports) for a hiring or \
talent-assessment need. It does not give general hiring/recruiting advice unrelated to \
test selection, does not give legal/compliance/regulatory advice, and does not do tasks \
unrelated to SHL assessments (general coding help, writing job descriptions from scratch, \
trivia, etc).

SHL test-type legend: A=Ability & Aptitude, B=Biodata & Situational Judgment, \
C=Competencies, D=Development & 360, E=Assessment Exercises, K=Knowledge & Skills, \
P=Personality & Behavior, S=Simulations.

Return JSON with exactly these fields:
{
  "in_scope": bool,                      // false if the LATEST user message is off-topic,
                                          // asks for legal/compliance/regulatory advice, or
                                          // is unrelated to SHL assessment selection
  "refusal_category": "off_topic" | "legal_advice" | null,
  "intent_compare": bool,                // true if the latest user message asks to compare
                                          // two or more specific named assessments
  "compare_names": [string],             // the assessment name(s)/fragments to compare, if any
  "constraints": {
    "role_or_skill": string|null,        // e.g. "Java developer", "sales", "Rust engineer"
    "seniority": string|null,            // e.g. "entry-level", "mid-level", "senior", "executive"
    "test_types": [string],              // letter codes the user explicitly wants, if stated
    "max_duration_minutes": int|null,
    "remote_required": bool|null,
    "language": string|null,
    "keywords": [string]                 // other relevant free-text keywords (domain, tools, traits)
  },
  "ready_to_recommend": bool,            // true if there's enough info to commit to a shortlist now
  "prior_recommendations_given": bool,   // true if an earlier assistant turn already showed a shortlist table
  "user_is_closing": bool,               // true if the latest user message is acceptance/closing
                                          // ("that works", "perfect", "keep as-is", "thanks") rather
                                          // than new information or a new question
  "clarifying_question": string|null     // a single natural follow-up question, if ready_to_recommend is false
}

Accumulate constraints across the ENTIRE conversation, not just the latest message — \
later messages refine or add to earlier ones (e.g. "actually add personality tests" \
should be merged into the existing constraints, not replace them, unless the user \
explicitly contradicts an earlier statement, in which case the later statement wins). \
If the user says they have no preference about something, leave that field null/empty \
rather than guessing."""


@dataclass
class Constraints:
    role_or_skill: str | None = None
    seniority: str | None = None
    test_types: list[str] = field(default_factory=list)
    max_duration_minutes: int | None = None
    remote_required: bool | None = None
    language: str | None = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class Interpretation:
    in_scope: bool = True
    refusal_category: str | None = None
    intent_compare: bool = False
    compare_names: list[str] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)
    ready_to_recommend: bool = False
    prior_recommendations_given: bool = False
    user_is_closing: bool = False
    clarifying_question: str | None = None
    llm_failed: bool = False


def _format_history(messages: list[ChatMessage]) -> str:
    lines = [f"{m.role.upper()}: {m.content}" for m in messages]
    return "\n".join(lines)


def _fallback_interpretation() -> Interpretation:
    """Used when the LLM call fails outright. Stays in-scope and asks a safe
    generic clarifying question rather than erroring out or guessing scope."""
    return Interpretation(
        in_scope=True,
        ready_to_recommend=False,
        clarifying_question=(
            "Could you tell me more about the role and what you'd like to assess "
            "(skills, personality, seniority level, etc.)?"
        ),
        llm_failed=True,
    )


def interpret_conversation(messages: list[ChatMessage]) -> Interpretation:
    raw = chat_json(SYSTEM_PROMPT, _format_history(messages))
    if raw is None:
        return _fallback_interpretation()

    try:
        c = raw.get("constraints", {}) or {}
        constraints = Constraints(
            role_or_skill=c.get("role_or_skill"),
            seniority=c.get("seniority"),
            test_types=[t for t in (c.get("test_types") or []) if isinstance(t, str)],
            max_duration_minutes=c.get("max_duration_minutes"),
            remote_required=c.get("remote_required"),
            language=c.get("language"),
            keywords=[k for k in (c.get("keywords") or []) if isinstance(k, str)],
        )
        return Interpretation(
            in_scope=bool(raw.get("in_scope", True)),
            refusal_category=raw.get("refusal_category"),
            intent_compare=bool(raw.get("intent_compare", False)),
            compare_names=[n for n in (raw.get("compare_names") or []) if isinstance(n, str)],
            constraints=constraints,
            ready_to_recommend=bool(raw.get("ready_to_recommend", False)),
            prior_recommendations_given=bool(raw.get("prior_recommendations_given", False)),
            user_is_closing=bool(raw.get("user_is_closing", False)),
            clarifying_question=raw.get("clarifying_question"),
        )
    except Exception:
        return _fallback_interpretation()
