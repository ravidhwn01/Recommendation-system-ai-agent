"""
Transforms the raw scraped catalog (catalog.Json) into data/catalog.json:
the cleaned, filtered, retrieval-ready Individual Test Solutions catalog.

Run:  python scripts/build_catalog.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "catalog_raw.json"
OUT_PATH = ROOT / "data" / "catalog.json"

# SHL's published test-type legend (https://www.shl.com/solutions/products/product-catalog/)
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Pre-packaged Job Solutions are bundled multi-test products (out of scope per the
# brief, which restricts the catalog to Individual Test Solutions). They are not
# tagged separately in the raw scrape, so they're identified by their "X Solution"
# naming convention, confirmed by manual inspection of all matches.
EXCLUDED_NAMES = {
    "Customer Service Phone Solution",
    "Entry Level Cashier Solution",
    "Entry Level Customer Service (General) Solution",
    "Entry Level Hotel Front Desk Solution",
    "Entry Level Sales Solution",
    "Entry Level Technical Support Solution",
    "Sales & Service Phone Solution",
}


def derive_test_type(keys: list[str]) -> str:
    codes = sorted({CATEGORY_TO_CODE[k] for k in keys if k in CATEGORY_TO_CODE})
    return ", ".join(codes)


def clean_item(item: dict) -> dict:
    return {
        "entity_id": item["entity_id"],
        "name": item["name"],
        "url": item["link"],
        "test_type": derive_test_type(item.get("keys", [])),
        "categories": item.get("keys", []),
        "description": item.get("description", ""),
        "job_levels": item.get("job_levels", []),
        "languages": item.get("languages", []),
        "duration": item.get("duration", ""),
        "remote": item.get("remote", ""),
        "adaptive": item.get("adaptive", ""),
    }


def main() -> None:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))

    seen_ids = set()
    cleaned = []
    for item in raw:
        if item["name"] in EXCLUDED_NAMES:
            continue
        if item["entity_id"] in seen_ids:
            continue
        seen_ids.add(item["entity_id"])
        cleaned.append(clean_item(item))

    cleaned.sort(key=lambda x: x["name"].lower())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"raw items:     {len(raw)}")
    print(f"excluded:      {len(raw) - len(cleaned)}")
    print(f"final catalog: {len(cleaned)} -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
