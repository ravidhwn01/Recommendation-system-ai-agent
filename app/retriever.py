"""
BM25 + metadata-filtered retrieval over the SHL Individual Test Solutions catalog.

The catalog is small (370 items) and queries are usually near-exact product names
("OPQ32r", "GSA") or domain keywords ("HIPAA", "Java", "sales") rather than long
natural-language questions, so BM25 keyword ranking is used instead of embeddings:
it handles exact/rare-term matches precisely, needs no model download, and is
deterministic. See README "Architecture" for the full rationale.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# How many times each field's tokens are repeated when building the BM25 document.
# This is the standard cheap field-boosting trick for rank_bm25, which has no
# native per-field weighting: name matches should dominate description matches.
FIELD_WEIGHTS = {
    "name": 4,
    "categories": 2,
    "description": 1,
    "job_levels": 1,
}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def parse_duration_minutes(duration: str) -> Optional[int]:
    match = re.search(r"\d+", duration or "")
    return int(match.group()) if match else None


def _document_tokens(item: dict) -> list[str]:
    tokens: list[str] = []
    tokens += tokenize(item["name"]) * FIELD_WEIGHTS["name"]
    tokens += tokenize(" ".join(item.get("categories", []))) * FIELD_WEIGHTS["categories"]
    tokens += tokenize(item.get("description", "")) * FIELD_WEIGHTS["description"]
    tokens += tokenize(" ".join(item.get("job_levels", []))) * FIELD_WEIGHTS["job_levels"]
    return tokens


@dataclass
class SearchFilters:
    test_types: Optional[set[str]] = None      # e.g. {"K", "P"} — item must match at least one
    job_levels: Optional[set[str]] = None       # item must match at least one
    max_duration_minutes: Optional[int] = None  # item duration must be <= this (unknown duration passes)
    remote_only: bool = False
    language: Optional[str] = None              # substring match against item languages


def item_matches_filters(item: dict, filters: SearchFilters, duration_minutes: Optional[int] = None) -> bool:
    """Standalone filter check reusable outside the BM25 hot path (e.g. for a
    specific name lookup, not just ranked search results). Pass duration_minutes
    to skip re-parsing the duration string when the caller already has it."""
    if filters.test_types:
        item_types = {t.strip() for t in item.get("test_type", "").split(",") if t.strip()}
        if not item_types & filters.test_types:
            return False

    if filters.job_levels:
        if not set(item.get("job_levels", [])) & filters.job_levels:
            return False

    if filters.max_duration_minutes is not None:
        mins = duration_minutes if duration_minutes is not None else parse_duration_minutes(item.get("duration", ""))
        if mins is not None and mins > filters.max_duration_minutes:
            return False

    if filters.remote_only and item.get("remote", "").lower() != "yes":
        return False

    if filters.language:
        lang_q = filters.language.lower()
        if not any(lang_q in lang.lower() for lang in item.get("languages", [])):
            return False

    return True


class CatalogIndex:
    def __init__(self, catalog_path: Path = DEFAULT_CATALOG_PATH):
        self.catalog: list[dict] = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self._corpus = [_document_tokens(item) for item in self.catalog]
        self._bm25 = BM25Okapi(self._corpus)
        self._duration_minutes = [parse_duration_minutes(item.get("duration", "")) for item in self.catalog]

    def _passes_filters(self, idx: int, filters: SearchFilters) -> bool:
        return item_matches_filters(self.catalog[idx], filters, duration_minutes=self._duration_minutes[idx])

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Returns up to top_k catalog items (with a `score` field), ranked by BM25
        relevance among items that pass `filters`. Hard filters are applied before
        ranking so they can never be crowded out by irrelevant high-scoring items."""
        filters = filters or SearchFilters()
        candidate_idx = [i for i in range(len(self.catalog)) if self._passes_filters(i, filters)]
        if not candidate_idx:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            ranked = candidate_idx
            scores = {i: 0.0 for i in candidate_idx}
        else:
            all_scores = self._bm25.get_scores(query_tokens)
            scores = {i: all_scores[i] for i in candidate_idx}
            ranked = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked[:top_k]:
            results.append({**self.catalog[i], "score": round(float(scores[i]), 4)})
        return results

    def find_by_name(self, name_fragment: str, top_k: int = 3) -> list[dict]:
        """Best-effort lookup of catalog items by (partial) product name — used for
        grounding comparison questions ("difference between OPQ and GSA") in real
        catalog records rather than the model's prior knowledge."""
        frag_tokens = set(tokenize(name_fragment))
        if not frag_tokens:
            return []

        scored = []
        for item in self.catalog:
            name_tokens = set(tokenize(item["name"]))
            overlap = len(frag_tokens & name_tokens)
            if overlap == 0:
                continue
            # exact substring match on the raw name is a strong signal (handles
            # alphanumeric product codes like "OPQ32r" that tokenize as one token)
            substring_bonus = 1 if name_fragment.lower() in item["name"].lower() else 0
            scored.append((overlap + substring_bonus, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]


_cached_index: Optional["CatalogIndex"] = None


def get_index(catalog_path: Path = DEFAULT_CATALOG_PATH) -> "CatalogIndex":
    """Process-wide cached index — building BM25 from scratch on every request
    would needlessly redo work that's identical across requests."""
    global _cached_index
    if _cached_index is None:
        _cached_index = CatalogIndex(catalog_path)
    return _cached_index
