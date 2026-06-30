from app.retriever import CatalogIndex, SearchFilters, parse_duration_minutes, tokenize


def index() -> CatalogIndex:
    return CatalogIndex()


def test_loads_full_catalog():
    idx = index()
    assert len(idx.catalog) == 370


def test_exact_product_name_ranks_first():
    idx = index()
    results = idx.search("OPQ32r", top_k=5)
    assert results
    assert "OPQ32r" in results[0]["name"]


def test_domain_keyword_surfaces_relevant_items():
    idx = index()
    results = idx.search("Java developer", top_k=10)
    names = [r["name"].lower() for r in results]
    assert any("java" in n for n in names)


def test_test_type_filter_excludes_other_types():
    idx = index()
    results = idx.search("assessment", filters=SearchFilters(test_types={"P"}), top_k=20)
    assert results
    for r in results:
        item_types = {t.strip() for t in r["test_type"].split(",")}
        assert "P" in item_types


def test_remote_only_filter():
    idx = index()
    results = idx.search("test", filters=SearchFilters(remote_only=True), top_k=20)
    assert all(r["remote"].lower() == "yes" for r in results)


def test_max_duration_filter_excludes_longer_known_durations():
    idx = index()
    results = idx.search("test", filters=SearchFilters(max_duration_minutes=10), top_k=50)
    for r in results:
        mins = parse_duration_minutes(r["duration"])
        if mins is not None:
            assert mins <= 10


def test_empty_filters_return_empty_when_no_candidates():
    idx = index()
    results = idx.search("test", filters=SearchFilters(test_types={"Z"}), top_k=10)
    assert results == []


def test_find_by_name_locates_known_products():
    idx = index()
    matches = idx.find_by_name("OPQ")
    assert any("OPQ" in m["name"] for m in matches)

    matches = idx.find_by_name("Global Skills Assessment")
    assert matches
    assert matches[0]["name"] == "Global Skills Assessment"


def test_tokenize_handles_product_codes():
    assert "opq32r" in tokenize("OPQ32r")
    assert "net" in tokenize(".NET Framework 4.5")


def test_results_capped_at_top_k():
    idx = index()
    results = idx.search("test", top_k=3)
    assert len(results) <= 3
