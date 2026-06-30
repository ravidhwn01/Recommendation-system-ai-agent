"""Thin Groq wrapper. Every call is narrow-purpose (interpret OR compose) and
fails soft: on any error this returns None so callers can fall back gracefully
instead of bubbling a 500 up to the evaluator."""
import json
import logging

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
    return _client


def chat_json(system: str, user: str, temperature: float = 0.0) -> dict | None:
    """Calls the LLM in JSON mode and returns the parsed object, or None on
    any failure (network, timeout, malformed JSON)."""
    try:
        resp = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        logger.exception("chat_json failed")
        return None


def chat_text(system: str, user: str, temperature: float = 0.3) -> str | None:
    """Calls the LLM for free-text generation. Returns None on failure."""
    try:
        resp = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        return resp.choices[0].message.content
    except Exception:
        logger.exception("chat_text failed")
        return None
