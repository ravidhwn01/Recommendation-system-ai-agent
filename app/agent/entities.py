from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryEntities:

    role: Optional[str] = None

    skill: Optional[str] = None

    duration: Optional[int] = None

    job_level: Optional[str] = None

    language: Optional[str] = None

    remote: Optional[bool] = None

    adaptive: Optional[bool] = None

    assessment_type: Optional[str] = None