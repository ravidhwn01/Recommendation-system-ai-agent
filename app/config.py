import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Per-call LLM timeout. The evaluator caps the whole /chat call at 30s and a turn
# uses at most two LLM calls (interpret + compose), so this leaves headroom for
# retrieval and network overhead.
LLM_TIMEOUT_SECONDS = 10.0

# Spec: evaluator caps each conversation at 8 turns including user & assistant
# messages, i.e. len(messages) must never be allowed to exceed this.
MAX_CONVERSATION_MESSAGES = 8

# Stop asking clarifying questions once the incoming history reaches this length
# (matches the observed pattern in the provided traces: at most 2 rounds of
# clarification before committing to a shortlist), so there's always room left
# before the hard cap.
MAX_MESSAGES_BEFORE_FORCED_COMMIT = 5
