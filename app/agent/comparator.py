import re

from app.data.schemas import expand_test_type
from app.rag.retriever import Retriever
from app.services.vector_service import VectorService


class ComparisonEngine:
    """Grounded comparison of two assessments using only catalog data."""

    def __init__(self):
        self.retriever = Retriever()
        self._metadatas = None

    @property
    def metadatas(self) -> list[dict]:
        if self._metadatas is None:
            self._metadatas = VectorService().get_all_metadatas()
        return self._metadatas

    def _summarize(self, meta: dict) -> str:
        parts = [f"- {meta.get('name', 'Unknown')} ({meta.get('url', '')})"]
        if meta.get("test_type"):
            readable = expand_test_type(meta["test_type"])
            label = f"{meta['test_type']} ({readable})" if readable \
                else meta["test_type"]
            parts.append(f"  Test type: {label}")
        if meta.get("duration"):
            parts.append(f"  Duration: {meta['duration']} minutes")
        if meta.get("job_levels"):
            parts.append(f"  Job levels: {meta['job_levels']}")
        if meta.get("languages"):
            parts.append(f"  Languages: {meta['languages']}")
        parts.append(
            f"  Remote testing: {'yes' if meta.get('remote_testing') else 'no'}"
        )
        parts.append(f"  Adaptive: {'yes' if meta.get('adaptive') else 'no'}")
        return "\n".join(parts)

    def _match_by_name(self, term: str) -> dict | None:
        """Exact/substring match on catalog names (grounds acronyms like OPQ)."""
        term = term.strip().lower()
        if len(term) < 2:
            return None
        matches = [
            m for m in self.metadatas
            if term in (m.get("name") or "").lower()
        ]
        if not matches:
            return None
        # Prefer the most specific (shortest) matching product name.
        return min(matches, key=lambda m: len(m.get("name") or ""))

    STOP = {
        "what", "whats", "what's", "is", "are", "the", "a", "an", "of", "do",
        "does", "how", "tell", "me", "please", "explain", "difference",
        "differences", "differ", "compare", "comparison", "between", "and",
        "or", "vs", "versus", "with", "for", "to", "in", "better", "which",
    }

    def _extract_terms(self, query: str) -> list[str]:
        text = re.sub(r"[?./]", " ", query.lower())

        parts = re.split(
            r"\bvs\.?\b|\bversus\b|\band\b|\bwith\b|\bbetween\b|,",
            text,
        )

        terms = []
        for part in parts:
            words = [w for w in part.split() if w not in self.STOP]
            term = " ".join(words).strip()
            if len(term) >= 2:
                terms.append(term)

        return terms

    def _pick_targets(self, query: str) -> list[dict]:
        terms = self._extract_terms(query)

        picked: list[dict] = []
        seen: set[str] = set()

        for term in terms:
            meta = self._match_by_name(term)
            if not meta:
                docs = self.retriever.search(term, k=1)
                meta = docs[0].metadata if docs else None
            if meta:
                name = meta.get("name")
                if name and name not in seen:
                    picked.append(meta)
                    seen.add(name)
            if len(picked) >= 2:
                break

        if len(picked) < 2:
            for doc in self.retriever.search(query, k=5):
                name = doc.metadata.get("name")
                if name and name not in seen:
                    picked.append(doc.metadata)
                    seen.add(name)
                if len(picked) >= 2:
                    break

        return picked[:2]

    def resolve_targets(self, query: str) -> list[dict]:
        """Public: return up to two grounded catalog metadata dicts to compare."""
        return self._pick_targets(query)

    def format_target(self, meta: dict) -> str:
        """Catalog data for one assessment, formatted for an LLM prompt."""
        return self._summarize(meta)

    def compare_reply(self, query: str) -> str:
        """Deterministic (non-LLM) grounded comparison; used as a fallback."""
        targets = self._pick_targets(query)

        if len(targets) < 2:
            return (
                "I couldn't find two matching SHL assessments to compare. "
                "Could you name the assessments you'd like compared?"
            )

        return (
            "Here's how these two SHL assessments compare, based on the "
            "catalog:\n\n"
            + self._summarize(targets[0])
            + "\n\n"
            + self._summarize(targets[1])
        )
