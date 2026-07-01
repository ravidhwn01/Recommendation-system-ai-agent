from typing import Optional

from pydantic import BaseModel, Field


class Assessment(BaseModel):

    id: Optional[str] = None

    name: str

    description: str = ""

    url: str = ""

    duration: str = ""

    remote_testing: str = ""

    adaptive: str = ""

    test_type: str = ""

    job_levels: list[str] = Field(default_factory=list)

    languages: list[str] = Field(default_factory=list)

    skills: str = ""