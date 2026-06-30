from app.policy import decide
from app.schemas import ChatMessage
from app.slots import Constraints, Interpretation


def msgs(n: int) -> list[ChatMessage]:
    """n alternating messages ending in a user turn, content irrelevant to policy."""
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append(ChatMessage(role=role, content=f"turn {i}"))
    return out


def test_refuses_when_out_of_scope():
    interp = Interpretation(in_scope=False, refusal_category="legal_advice")
    decision = decide(msgs(1), interp)
    assert decision.action == "refuse"
    assert decision.refusal_category == "legal_advice"
    assert decision.end_of_conversation is False


def test_refusal_does_not_end_conversation_below_cap():
    interp = Interpretation(in_scope=False, refusal_category="off_topic")
    decision = decide(msgs(5), interp)
    assert decision.action == "refuse"
    assert decision.end_of_conversation is False


def test_refusal_ends_conversation_at_turn_cap():
    interp = Interpretation(in_scope=False, refusal_category="off_topic")
    decision = decide(msgs(7), interp)
    assert decision.end_of_conversation is True


def test_clarifies_when_vague_and_under_budget():
    interp = Interpretation(in_scope=True, ready_to_recommend=False)
    decision = decide(msgs(1), interp)
    assert decision.action == "clarify"
    assert decision.end_of_conversation is False


def test_forces_commit_after_two_clarify_rounds_even_if_not_ready():
    interp = Interpretation(in_scope=True, ready_to_recommend=False)
    decision = decide(msgs(5), interp)
    assert decision.action == "answer"


def test_recommends_when_ready():
    interp = Interpretation(
        in_scope=True,
        ready_to_recommend=True,
        constraints=Constraints(role_or_skill="Java developer"),
    )
    decision = decide(msgs(3), interp)
    assert decision.action == "answer"
    assert decision.end_of_conversation is False  # not closing yet


def test_ends_conversation_when_user_is_closing():
    interp = Interpretation(in_scope=True, prior_recommendations_given=True, user_is_closing=True)
    decision = decide(msgs(4), interp)
    assert decision.action == "answer"
    assert decision.end_of_conversation is True


def test_comparison_request_triggers_answer_even_if_not_ready():
    interp = Interpretation(
        in_scope=True,
        ready_to_recommend=False,
        intent_compare=True,
        compare_names=["OPQ32r", "GSA"],
    )
    decision = decide(msgs(1), interp)
    assert decision.action == "answer"
    assert decision.comparison_requested is True
    assert decision.compare_names == ["OPQ32r", "GSA"]


def test_retrieval_query_built_from_constraints():
    interp = Interpretation(
        in_scope=True,
        ready_to_recommend=True,
        constraints=Constraints(role_or_skill="Java developer", seniority="mid-level", keywords=["stakeholders"]),
    )
    decision = decide(msgs(3), interp)
    assert "Java developer" in decision.retrieval_query
    assert "mid-level" in decision.retrieval_query
    assert "stakeholders" in decision.retrieval_query


def test_filters_built_from_explicit_test_type_constraint():
    interp = Interpretation(
        in_scope=True,
        ready_to_recommend=True,
        constraints=Constraints(role_or_skill="manager", test_types=["P"]),
    )
    decision = decide(msgs(3), interp)
    assert decision.filters.test_types == {"P"}
