import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.dataset_service import DatasetService


async def _load_assessments_count() -> tuple[int, object]:

    service = DatasetService()

    assessments = await service.load_assessments()

    return len(assessments), assessments[0]


def test_loader_reads_catalog() -> None:

    count, first = asyncio.run(_load_assessments_count())

    assert count > 0
    assert first is not None


if __name__ == "__main__":
    count, first = asyncio.run(_load_assessments_count())
    print(count)
    print(first)