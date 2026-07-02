import re

from app.agent.entities import QueryEntities


class EntityExtractor:

    def extract(
        self,
        query: str,
    ) -> QueryEntities:

        entities = QueryEntities()

        text = query.lower()

        if "java" in text:
            entities.skill = "Java"

        elif "python" in text:
            entities.skill = "Python"

        elif "sql" in text:
            entities.skill = "SQL"

        if "graduate" in text:
            entities.job_level = "Graduate"

        elif "manager" in text:
            entities.job_level = "Manager"

        if "remote" in text:
            entities.remote = True

        duration = re.search(
            r"(\d+)\s*(minute|min)",
            text,
        )

        if duration:
            entities.duration = int(
                duration.group(1)
            )

        return entities