"""FastAPI service: GET /health, POST /chat. The response schema is exact per
spec — see app/schemas.py — and /chat is stateless: every call reconstructs
state from the full message history via app/pipeline.py."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.config import CHAT_HARD_DEADLINE_SECONDS
from app.pipeline import handle_chat
from app.retriever import get_index
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = ChatResponse(
    reply="I'm having trouble processing that right now - could you try again?",
    recommendations=[],
    end_of_conversation=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_index()  # build the BM25 index once at startup, not on the first request
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await asyncio.wait_for(
            run_in_threadpool(handle_chat, request.messages),
            timeout=CHAT_HARD_DEADLINE_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("handle_chat exceeded the hard deadline")
        return FALLBACK_RESPONSE
    except Exception:
        logger.exception("handle_chat failed")
        return FALLBACK_RESPONSE
