from app.compose import compose_answer
from app.pipeline import DEFAULT_PERSONALITY_ASSESSMENT_NAME, _retrieve_candidates, _supplement_with_personality
from app.policy import PolicyDecision
from app.retriever import SearchFilters, get_index


def _decision(query: str, test_types: set[str]) -> PolicyDecision:
    return PolicyDecision(
        action="answer",
        end_of_conversation=False,
        refusal_category=None,
        comparison_requested=False,
        compare_names=[],
        should_recommend=True,
        force_commit=False,
        retrieval_query=query,
        filters=SearchFilters(test_types=test_types),
        clarifying_question=None,
    )


def test_multi_type_query_does_not_starve_minority_type():
    """A role-biased query ("Java developer...") must not let the dominant
    test_type crowd out a minority type the user explicitly asked to add -
    regression test for the bug where personality-type items never reached
    the candidate pool for a heavily technical query."""
    index = get_index()
    decision = _decision(
        "Java developer mid-level stakeholders technical skills collaboration",
        {"K", "P"},
    )
    results = _retrieve_candidates(index, decision)

    types_present = {t.strip() for r in results for t in r["test_type"].split(",")}
    assert "K" in types_present
    assert "P" in types_present


def test_single_type_query_unaffected():
    index = get_index()
    decision = _decision("Java developer", {"K"})
    results = _retrieve_candidates(index, decision)
    for r in results:
        item_types = {t.strip() for t in r["test_type"].split(",")}
        assert "K" in item_types


def test_no_type_filter_returns_full_pool():
    index = get_index()
    decision = _decision("Java developer", set())
    results = _retrieve_candidates(index, decision)
    assert len(results) <= 20
    assert results  # non-empty for a query that matches real catalog items


def test_personality_supplement_is_prepended_not_appended():
    """Regression test for a bug found via eval/run_eval.py: compose.py's
    graceful-degradation fallback takes candidates[:5] when the LLM call
    itself fails. If the personality supplement were appended at the end of
    a long candidate list (as it originally was), that fallback would slice
    it off completely - silently dropping OPQ32r in exactly the rate-limited/
    degraded scenario the supplement matters most for."""
    index = get_index()
    decision = _decision("Java Spring SQL AWS Docker", {"K"})
    candidates = _retrieve_candidates(index, decision)
    supplemented = _supplement_with_personality(index, decision, candidates)

    assert supplemented[0]["name"] == DEFAULT_PERSONALITY_ASSESSMENT_NAME
    assert any(c["name"] == DEFAULT_PERSONALITY_ASSESSMENT_NAME for c in supplemented[:5])


def test_compose_fallback_includes_personality_supplement_on_llm_failure(monkeypatch):
    """End-to-end check (LLM mocked to fail) that the fallback path itself -
    not just candidate ordering - actually surfaces OPQ32r when present."""
    monkeypatch.setattr("app.compose.chat_json", lambda *a, **k: None)

    index = get_index()
    decision = _decision("Java Spring SQL AWS Docker", {"K"})
    decision.force_commit = True
    candidates = _retrieve_candidates(index, decision)
    candidates = _supplement_with_personality(index, decision, candidates)

    _, recommendations = compose_answer(decision, candidates, [], "role: Java developer")
    assert any(r.name == DEFAULT_PERSONALITY_ASSESSMENT_NAME for r in recommendations)
