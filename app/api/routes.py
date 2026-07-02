from fastapi import APIRouter

from app.agent.agent import Agent
from app.models.chat import ChatRequest
from app.models.response import ChatResponse

router = APIRouter()

agent = Agent()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    messages = [m.model_dump() for m in request.messages]
    return agent.handle(messages)
