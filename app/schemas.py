"""Pydantic request/response models. The shape here is non-negotiable per the
spec — it is what the automated evaluator parses on every /chat call."""
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    end_of_conversation: bool = False
