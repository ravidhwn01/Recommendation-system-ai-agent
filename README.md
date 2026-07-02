# SHL Conversational Assessment Recommender

A conversational agent that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a grounded shortlist of SHL assessments,
through dialogue — served over a stateless FastAPI service.

It clarifies vague requests, recommends 1–10 assessments with real catalog
URLs, refines the shortlist when constraints change, compares assessments from
catalog data, and refuses anything out of scope.

---

## How it works

There are two workflows: an offline build that turns the raw catalog into a
searchable vector store, and the online request flow that answers `/chat`.

### 1. Offline: build the vector store

```
data/raw/catalog_raw.json            (377 scraped SHL catalog items)
        │
        ▼
DataLoader → DatasetValidator → DataCleaner        (app/data/)
        │   normalize each record into an Assessment:
        │   link→url, keys→test_type code (A/B/C/D/E/K/P/S),
        │   remote/adaptive→bool, duration→int, entity_id→id
        ▼
DocumentBuilder                                     (app/rag/)
        │   one text document per assessment + metadata
        │   (name, url, test_type, duration, job_levels, …)
        ▼
Embeddings (BAAI/bge-small-en-v1.5) → Chroma        (vector_db/)
```

Run once with `python build_vector_db.py`. URLs and test-types live in the
document metadata, so anything recommended later is grounded in the catalog.

### 2. Online: a `/chat` turn

The API is stateless — every call carries the full message history. The agent
([app/agent/agent.py](app/agent/agent.py)) rebuilds context from that history
each time:

```
POST /chat  (full conversation history)
        │
        ▼
DECISION  (1 LLM call, JSON)  →  clarify | recommend | refine | compare | refuse
        │                        + a distilled search query
        ├── clarify ─────────►  ask ONE targeted question, no recommendations
        ├── refuse ──────────►  fixed in-scope refusal, no recommendations
        │
        ├── recommend/refine ►  retrieve ~20 candidates from Chroma
        │                        → SELECT (1 LLM call): pick the best 1–10 by index
        │                        → build recommendations from catalog metadata
        │                          (name / url / test_type taken verbatim)
        │
        └── compare ─────────►  resolve the two named assessments (name match +
                                 embeddings) → grounded LLM comparison
```

At most 2 LLM calls per turn (decision + selection); clarify and refuse use
none. Typical latency ~1s, well inside the 30s cap.

### The four behaviors

- Clarify — a vague query ("I need an assessment") or a named role missing
  key detail ("a Java developer") gets one targeted question before any
  shortlist. It commits within the turn budget instead of asking forever.
- Recommend — once there's enough context (role + detail, or a pasted job
  description), it returns 1–10 assessments with names and catalog URLs.
- fine — "actually, add a personality test" re-retrieves for the new
  constraint and updates the existing shortlist rather than starting over.
- mpare — "difference between OPQ and GSA" is answered from the two
  catalog records' fields, not the model's prior knowledge.

### Grounding & safety

- URLs never hallucinated — the LLM only *selects* from retrieved
  candidates; `name`/`url`/`test_type` are copied from catalog metadata.
- Refusals are fixed text — off-topic, legal/HR advice, and prompt
  injection get a canned reply the model can't be talked out of.
- Graceful fallback — every LLM call has a timeout; on failure the agent
  falls back to a rule-based decision + plain retrieval, so `/chat` never 500s.
- `end_of_conversation` is `true` when a shortlist is committed, `false`
  while clarifying, comparing, or refusing.

---

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI + Pydantic v2 |
| Vector store | Chroma (persisted in `vector_db/`) |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, CPU) |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Orchestration | LangChain (documents + Chroma) |
| Logging | loguru |

---

## Running locally

```bash
# 1. install deps (into your virtualenv)
pip install -r requirements.txt

# 2. set secrets in .env
#    GROQ_API_KEY=...   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
#    VECTOR_DB_PATH=./vector_db   DATA_PATH=./data

# 3. build the vector store (once)
python build_vector_db.py

# 4. run the service
uvicorn app.main:app --port 8000
```

Endpoints:

- `GET /health` → `{"status": "ok"}`
- `POST /chat`

```jsonc
// request
{ "messages": [ { "role": "user", "content": "Hiring a mid-level Java developer" } ] }

// response
{
  "reply": "Here are assessments that fit a mid-level Java developer.",
  "recommendations": [
    { "name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K" }
  ],
  "end_of_conversation": true
}
```

`recommendations` is empty while clarifying or refusing, and a 1–10 item array
once a shortlist is committed.

---

## Project layout

```
app/
├── main.py                 FastAPI app (/health, /chat)
├── api/routes.py           thin endpoint → Agent
├── agent/
│   ├── agent.py            orchestration: decide → recommend/refine/compare/clarify/refuse
│   ├── prompt_builder.py   LLM prompts (decision, selection, comparison)
│   ├── comparator.py       grounded two-assessment comparison
│   ├── context.py          pull latest / all user text from history
│   ├── guardrails.py       injection + off-topic detection (fallback)
│   └── intent.py           keyword intent (fallback)
├── rag/
│   ├── document_builder.py Assessment → LangChain Document + metadata
│   ├── embedding.py        HuggingFace embedding model
│   └── retriever.py        similarity search over Chroma
├── data/
│   ├── loader.py           read raw catalog JSON
│   ├── schemas.py          Assessment model + raw→clean mapping/coercion
│   ├── validator.py        validate records into Assessments
│   └── cleaner.py          whitespace cleanup
├── services/
│   ├── llm_service.py      Groq wrapper (generate_json / generate_text, fail-soft)
│   └── vector_service.py   Chroma access (search, get_all_metadatas, add)
├── models/                 request/response Pydantic schemas
└── core/                   config + logger

build_vector_db.py          build the Chroma store from the raw catalog
data/raw/catalog_raw.json    377 scraped SHL catalog items
vector_db/                   persisted Chroma store (generated)
```
