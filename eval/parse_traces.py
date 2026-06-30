"""Parses the provided eval/traces/*.md conversation traces into structured
TraceCase objects: the scripted sequence of user messages, plus the labeled
expected shortlist (the table on the turn where end_of_conversation: true).

This is a *scripted replay* harness, not a full user-simulation harness: the
real evaluator drives an LLM-simulated user persona that adapts to our agent's
actual replies, whereas this replays the trace's exact recorded user messages
regardless of what our agent says back. That's a reasonable, fast, deterministic
proxy for local development/tuning (see README "Phase 4 findings" for why), but
it cannot perfectly reproduce the real evaluator's grading.
"""
import re
from dataclasses import dataclass
from pathlib import Path

TRACES_DIR = Path(__file__).resolve().parent / "traces"

_TURN_RE = re.compile(r"### Turn \d+\n(.*?)(?=### Turn \d+|\Z)", re.DOTALL)
_USER_RE = re.compile(r"\*\*User\*\*\s*\n+((?:^>.*\n?)+)", re.MULTILINE)
_END_RE = re.compile(r"end_of_conversation.*?\*\*(true|false)\*\*", re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|(.+)\|\s*$", re.MULTILINE)


@dataclass
class ExpectedItem:
    name: str
    test_type: str
    url: str


@dataclass
class TraceCase:
    trace_id: str
    user_turns: list[str]
    expected_shortlist: list[ExpectedItem]


def _extract_user_message(turn_text: str) -> str | None:
    m = _USER_RE.search(turn_text)
    if not m:
        return None
    lines = [line[1:].strip() for line in m.group(1).strip().splitlines()]
    return " ".join(line.lstrip(">").strip() for line in lines if line.strip())


def _extract_end_of_conversation(turn_text: str) -> bool:
    m = _END_RE.search(turn_text)
    return bool(m and m.group(1).lower() == "true")


def _extract_table(turn_text: str) -> list[ExpectedItem]:
    items = []
    for row_match in _TABLE_ROW_RE.finditer(turn_text):
        cols = [c.strip() for c in row_match.group(1).split("|")]
        if len(cols) < 6:
            continue
        name, test_type = cols[0], cols[1]
        url_col = cols[-1]
        url_match = re.search(r"<([^>]+)>", url_col)
        url = url_match.group(1) if url_match else url_col.strip()
        items.append(ExpectedItem(name=name, test_type=test_type, url=url))
    return items


def parse_trace_file(path: Path) -> TraceCase:
    text = path.read_text(encoding="utf-8")
    user_turns: list[str] = []
    expected_shortlist: list[ExpectedItem] = []

    for turn_match in _TURN_RE.finditer(text):
        turn_text = turn_match.group(1)

        user_msg = _extract_user_message(turn_text)
        if user_msg:
            user_turns.append(user_msg)

        if _extract_end_of_conversation(turn_text):
            expected_shortlist = _extract_table(turn_text)

    return TraceCase(trace_id=path.stem, user_turns=user_turns, expected_shortlist=expected_shortlist)


def load_all_traces() -> list[TraceCase]:
    return [parse_trace_file(p) for p in sorted(TRACES_DIR.glob("*.md"))]


if __name__ == "__main__":
    for case in load_all_traces():
        print(f"{case.trace_id}: {len(case.user_turns)} user turns, "
              f"{len(case.expected_shortlist)} expected items")
        for item in case.expected_shortlist:
            print(f"   - {item.name} ({item.test_type})")
