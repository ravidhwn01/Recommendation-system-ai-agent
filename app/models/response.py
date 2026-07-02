from pydantic import BaseModel

from app.models.recommendation import Recommendation


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool = False