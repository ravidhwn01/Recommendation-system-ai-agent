"""Thin Groq wrapper. Every call is narrow-purpose (interpret OR compose) and
fails soft: on any error this returns None so callers can fall back gracefully
instead of bubbling a 500 up to the evaluator."""
import json
import logging
import time

from groq import Groq, RateLimitError

from app.config import GROQ_API_KEY, GROQ_MODEL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_client: Groq | None = None

# A 429 is recoverable and worth one short, capped retry - unlike other failure
# modes, retrying blindly here would risk compounding past the /chat hard
# deadline, so this is deliberately a single bounded backoff, not the SDK's
# own (disabled) retry behavior.
RATE_LIMIT_BACKOFF_SECONDS = 3.0


def _get_client() -> Groq:
    global _client
    if _client is None:
        # max_retries=0: the SDK retries transient errors by default, which could
        # silently multiply a single call's worst-case latency past the timeout
        # and blow the /chat hard deadline. Failing fast (with one deliberate,
        # bounded retry for rate limits below) is more predictable under a
        # strict time budget.
        _client = Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS, max_retries=0)
    return _client


def _create(**kwargs):
    try:
        return _get_client().chat.completions.create(**kwargs)
    except RateLimitError:
        logger.warning("rate limited, retrying once after backoff")
        time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
        return _get_client().chat.completions.create(**kwargs)


def chat_json(system: str, user: str, temperature: float = 0.0) -> dict | None:
    """Calls the LLM in JSON mode and returns the parsed object, or None on
    any failure (network, timeout, malformed JSON, rate limit after retry)."""
    try:
        resp = _create(
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
        resp = _create(
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
