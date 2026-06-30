from app.pipeline import _retrieve_candidates
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
