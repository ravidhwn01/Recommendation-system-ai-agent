import os

from dotenv import load_dotenv

load_dotenv()

# The eval harness hit Groq's free-tier TPM rate limit partway through a
# 10-trace run. Counter to the usual assumption that smaller models get a
# higher throughput allowance, this account's free tier actually caps
# llama-3.1-8b-instant at a LOWER limit (6000 TPM) than llama-3.3-70b-versatile
# (12000 TPM) - confirmed empirically (see README "Phase 4 findings"), not
# assumed. Sticking with the 70b model; the real fix for rate-limit headroom is
# the prompt-size reduction in app/compose.py and the retry/backoff in
# app/llm.py, not a model swap. Override via env var if your account differs.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Per-call LLM timeout, with retries disabled (see app/llm.py). The evaluator caps
# the whole /chat call at 30s and a turn uses at most two LLM calls (interpret +
# compose) = 16s worst case, leaving headroom for retrieval, network overhead, and
# the hard deadline enforced in app/main.py.
LLM_TIMEOUT_SECONDS = 8.0

# /chat's own safety-net deadline (app/main.py) — if the pipeline runs past this,
# return a graceful fallback instead of risking the evaluator's 30s cap entirely.
CHAT_HARD_DEADLINE_SECONDS = 25.0

# Spec: evaluator caps each conversation at 8 turns including user & assistant
# messages, i.e. len(messages) must never be allowed to exceed this.
MAX_CONVERSATION_MESSAGES = 8

# Stop asking clarifying questions once the incoming history reaches this length
# (matches the observed pattern in the provided traces: at most 2 rounds of
# clarification before committing to a shortlist), so there's always room left
# before the hard cap.
MAX_MESSAGES_BEFORE_FORCED_COMMIT = 5
