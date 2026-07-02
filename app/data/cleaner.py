import re

from app.data.schemas import Assessment


class DataCleaner:

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean(self, assessments: list[Assessment]) -> list[Assessment]:

        cleaned = []

        for assessment in assessments:

            assessment.name = self.clean_text(assessment.name)

            assessment.description = self.clean_text(assessment.description)

            cleaned.append(assessment)

        return cleaned
