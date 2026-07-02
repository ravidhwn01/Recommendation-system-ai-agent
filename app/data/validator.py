from app.core.logger import logger
from app.data.schemas import Assessment


class DatasetValidator:

    def validate(self, records: list[dict]) -> list[Assessment]:

        validated = []

        skipped = 0

        for record in records:

            try:
                validated.append(Assessment.model_validate(record))

            except Exception as e:
                skipped += 1
                logger.warning(e)

        logger.info(f"Validated : {len(validated)}")

        logger.info(f"Skipped : {skipped}")

        return validated
