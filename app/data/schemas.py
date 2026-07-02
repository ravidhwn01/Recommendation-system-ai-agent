from typing import Optional

from pydantic import BaseModel, Field, model_validator

# SHL catalog "keys" categories -> single-letter test-type codes.
TEST_TYPE_CODES = {
    "ability & aptitude": "A",
    "biodata & situational judgment": "B",
    "biodata & situational judgement": "B",
    "competencies": "C",
    "development & 360": "D",
    "assessment exercises": "E",
    "knowledge & skills": "K",
    "personality & behavior": "P",
    "personality & behaviour": "P",
    "simulations": "S",
}

_TRUE_VALUES = {"yes", "true", "1", "y"}
_DURATION_NULLS = {"", "-", "n/a", "variable", "untimed", "tbc", "none"}


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUE_VALUES


def _to_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_duration(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _DURATION_NULLS:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def _keys_to_test_type(keys) -> str:
    codes = []
    for key in _to_list(keys):
        code = TEST_TYPE_CODES.get(key.strip().lower())
        if code and code not in codes:
            codes.append(code)
    return "".join(sorted(codes))


class Assessment(BaseModel):
    id: Optional[str] = None

    name: str

    description: str = ""

    url: str = ""

    duration: Optional[int] = None

    remote_testing: bool = False

    adaptive: bool = False

    test_type: str = ""

    job_levels: list[str] = Field(default_factory=list)

    languages: list[str] = Field(default_factory=list)

    skills: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _map_raw_catalog(cls, data):
        """Normalize a raw SHL catalog record into the clean schema.

        The scraped catalog uses different field names/types than the schema
        (link->url, keys->test_type, remote/adaptive as "yes"/"no", duration as
        free text, entity_id->id). This runs before validation so both raw
        catalog dicts and already-clean dicts load correctly.
        """
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # id
        if not d.get("id") and d.get("entity_id") is not None:
            d["id"] = str(d["entity_id"])

        # url
        if not d.get("url") and d.get("link"):
            d["url"] = d["link"]

        # test_type from catalog "keys" categories
        if not d.get("test_type") and d.get("keys") is not None:
            d["test_type"] = _keys_to_test_type(d["keys"])

        # remote_testing from "remote"
        if "remote_testing" not in d and "remote" in d:
            d["remote_testing"] = d["remote"]
        d["remote_testing"] = _to_bool(d.get("remote_testing"))

        # adaptive stays as-is name, coerce yes/no
        d["adaptive"] = _to_bool(d.get("adaptive"))

        # duration: prefer explicit numeric duration, fall back to duration_raw
        d["duration"] = _parse_duration(
            d.get("duration") if str(d.get("duration") or "").strip()
            else d.get("duration_raw")
        )

        # list fields
        d["job_levels"] = _to_list(d.get("job_levels"))
        d["languages"] = _to_list(d.get("languages"))
        d["skills"] = _to_list(d.get("skills"))

        # trim strings
        for key in ("name", "description", "url"):
            if isinstance(d.get(key), str):
                d[key] = d[key].strip()

        return d
