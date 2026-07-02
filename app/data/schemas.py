from pydantic import BaseModel, Field
from typing import Optional


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