from pydantic import BaseModel


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str