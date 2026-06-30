from app.compose import compose_answer
from app.policy import PolicyDecision
from app.retriever import SearchFilters

CANDIDATES = [
    {"entity_id": "1", "name": "Core Java (Advanced Level)", "url": "https://x/1", "test_type": "K",
     "duration": "30 minutes", "description": "Java knowledge test"},
    {"entity_id": "2", "name": "OPQ32r", "url": "https://x/2", "test_type": "P",
     "duration": "25 minutes", "description": "Personality questionnaire"},
]


def _decision(should_recommend: bool, force_commit: bool = False) -> PolicyDecision:
    return PolicyDecision(
        action="answer",
        end_of_conversation=False,
        refusal_category=None,
        comparison_requested=False,
        compare_names=[],
        should_recommend=should_recommend,
        force_commit=force_commit,
        retrieval_query="java developer",
        filters=SearchFilters(),
        clarifying_question=None,
    )


def test_should_recommend_true_never_returns_empty_when_llm_picks_nothing(monkeypatch):
    """Spec: recommendations is empty only when gathering context or refusing,
    and 1-10 items once committed - there's no valid 'committed but zero
    items' state. should_recommend=True means policy already decided to
    commit, so even if the LLM (legitimately or not) returns no indices, the
    response must still contain at least one item rather than violate the
    schema."""
    monkeypatch.setattr("app.compose.chat_json", lambda *a, **k: {"selected_indices": [], "reply": "ok"})
    _, recommendations = compose_answer(_decision(should_recommend=True), CANDIDATES, [], "role: Java developer")
    assert len(recommendations) >= 1


def test_should_recommend_false_stays_empty_even_if_llm_picks_something(monkeypatch):
    """The inverse guarantee: should_recommend=False must always yield an
    empty list, even if the LLM (incorrectly) returns indices - this is the
    'still gathering context / answering a standalone comparison' state."""
    monkeypatch.setattr("app.compose.chat_json", lambda *a, **k: {"selected_indices": [0, 1], "reply": "ok"})
    _, recommendations = compose_answer(_decision(should_recommend=False), CANDIDATES, [], "no constraints yet")
    assert recommendations == []


def test_llm_failure_with_should_recommend_true_falls_back_to_nonempty(monkeypatch):
    monkeypatch.setattr("app.compose.chat_json", lambda *a, **k: None)
    _, recommendations = compose_answer(_decision(should_recommend=True), CANDIDATES, [], "role: Java developer")
    assert len(recommendations) >= 1


def test_recommendations_never_exceed_ten_items(monkeypatch):
    many = [{**CANDIDATES[0], "entity_id": str(i), "url": f"https://x/{i}"} for i in range(15)]
    monkeypatch.setattr(
        "app.compose.chat_json",
        lambda *a, **k: {"selected_indices": list(range(15)), "reply": "ok"},
    )
    _, recommendations = compose_answer(_decision(should_recommend=True), many, [], "role: Java developer")
    assert len(recommendations) <= 10
