from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger
from app.data.schemas import Assessment


class DatasetService:

    def __init__(self) -> None:

        self.dataset_path = (
            Path(settings.DATA_PATH)
            / "raw"
            / "catalog.json"
        )

    async def load_assessments(self) -> list[Assessment]:

        logger.info("Loading SHL dataset...")

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"{self.dataset_path} does not exist."
            )

        with self.dataset_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            raw_data = json.load(file)

        assessments = []

        skipped = 0

        for row in raw_data:

            try:

                assessment = self._normalize(row)

                assessments.append(assessment)

            except Exception as e:

                skipped += 1

                logger.warning(e)

        logger.info(
            f"Loaded {len(assessments)} assessments "
            f"(Skipped {skipped})"
        )

        return assessments

    def _normalize(
        self,
        row: dict,
    ) -> Assessment:

        return Assessment(

            id=row.get("id"),

            name=row.get("name", "").strip(),

            description=row.get(
                "description",
                ""
            ).strip(),

            url=row.get("url", "").strip(),

            duration=self._parse_duration(
                row.get("duration")
            ),

            remote_testing=self._to_bool(
                row.get("remote_testing")
            ),

            adaptive=self._to_bool(
                row.get("adaptive")
            ),

            test_type=row.get(
                "test_type",
                ""
            ).strip(),

            job_levels=self._to_list(
                row.get("job_levels")
            ),

            languages=self._to_list(
                row.get("languages")
            ),

            skills=self._to_list(
                row.get("skills")
            ),

        )

    @staticmethod
    def _to_list(value):

        if value is None:
            return []

        if isinstance(value, list):
            return value

        if value == "":
            return []

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    @staticmethod
    def _to_bool(value):

        if value is None:
            return False

        if isinstance(value, bool):
            return value

        value = str(value).lower()

        return value in {
            "yes",
            "true",
            "1",
        }

    @staticmethod
    def _parse_duration(value):

        if value is None:
            return None

        value = str(value).lower()

        if value in {
            "",
            "-",
            "n/a",
            "variable",
            "untimed",
            "tbc",
        }:
            return None

        digits = "".join(
            ch
            for ch in value
            if ch.isdigit()
        )

        if digits == "":
            return None

        return int(digits)