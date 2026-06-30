# SHL Conversational Assessment Recommender

A conversational agent that takes a hiring manager from a vague intent ("I'm hiring a Java developer")
to a grounded shortlist of SHL Individual Test Solutions, via a stateless FastAPI service.

This README is also the **running project log** — it documents the plan, the decisions made (and why),
and what's still outstanding, so the design can be defended in the technical deep-dive interview.

---

## 1. Problem

Build an agent + API that:

- Indexes the SHL product catalog (Individual Test Solutions only — Job Solutions are out of scope).
- Exposes `GET /health` and `POST /chat`, where `/chat` is **stateless**: every call carries the full
  conversation history and must be re-derived from scratch (no server-side session state).
- Clarifies vague queries, recommends 1–10 assessments with `{name, url, test_type}`, refines the
  shortlist on changed constraints, and answers comparison questions grounded in catalog data.
- Stays in scope: refuses general hiring/legal advice and prompt-injection attempts.
- Is graded on: hard schema/catalog/turn-cap compliance, mean Recall@10 across conversation traces
  (public + holdout), and a behavior-probe pass rate (refusals, no-day-1-recommend, refinement honored,
  hallucination rate).

Full original brief: see [`docs/assignment.md`](docs/assignment.md) *(to be added)*.

---

## 2. Architecture

A single LLM "decide everything" loop is exactly what the brief warns against (non-deterministic
conversation breaking the system). Instead this is a **compound system**: deterministic Python control
flow wraps narrow, single-purpose LLM calls, so failures are bounded and debuggable.

```
POST /chat
   │
   ▼
[Guard layer]       off-topic / legal-advice / prompt-injection detection → refuse, return early
   │
   ▼
[Slot extractor]    re-derive state from full history every call (role, seniority, skills,
                     test-type prefs, duration/remote constraints, compare-intent)
   │
   ▼
[Policy]            deterministic logic: clarify | retrieve+recommend | refine | compare
                     (turn-budget aware — must commit to a shortlist before the 8-turn cap)
   ▼
[Retriever]         BM25 + metadata filters over catalog.json — LLM only ever selects from
                     retrieved candidates, never invents a name/URL
   │
   ▼
[Response composer] LLM writes reply text; `recommendations` array is built in Python from
                     catalog records verbatim, then schema-validated before returning
```

**Why not embeddings-first retrieval?** The catalog is a few hundred structured items, often queried
by near-exact name ("OPQ32r", "GSA"). BM25/TF-IDF over name + description + test-type, combined with
metadata filters, is expected to outperform embeddings here and avoids shipping a model file (keeps
cold start fast on free hosting tiers). Embeddings are a fallback if eval recall is insufficient.

**Why not a stateful service?** The spec requires statelessness — every `/chat` call gets the full
message history and must reconstruct slot-state, current shortlist, and turn count purely from that
history. No DB/session store is used or needed.

---

## 3. Tech stack

| Layer | Choice | Reason |
|---|---|---|
| API | FastAPI + Pydantic v2 | Exact schema validation, async, free-tier friendly |
| Retrieval | `rank_bm25` + metadata filters | Lightweight, no model download, fits structured catalog |
| LLM | Groq (Llama 3.x) | Free tier, low latency — fits the 30s/call budget |
| Scraping | requests + BeautifulSoup (Playwright fallback if catalog is JS-rendered) | TBD pending inspection of the live page |
| Deployment | TBD (Render / Fly.io / HF Spaces) | Dockerized so the choice stays late-binding |
| Eval | Custom harness over provided conversation traces + pytest | Needed to tune Recall@10 and verify hard-eval/behavior-probe compliance before submission |

---

## 4. Guardrails

- **Schema lock** — `recommendations` is always empty or 1–10 items, built only from catalog rows.
- **Scope refusal** — keyword + LLM-classifier guard for off-topic/legal/injection attempts; system
  instructions are never echoed back.
- **Turn-budget logic** — policy forces a best-effort shortlist before the 8-turn cap instead of
  clarifying indefinitely.
- **Refinement vs. restart** — slot-state is rebuilt from re-extracted history each call, so a
  constraint change updates the existing shortlist rather than discarding context.
- **Comparison grounding** — compare-intent pulls the two catalog records' real fields into the
  prompt; the LLM may not answer from prior/world knowledge.
- **Graceful degradation** — LLM timeout/error falls back to pure retrieval rather than a 500.

---

## 5. Delivery phases

| Phase | Output | Status |
|---|---|---|
| 0 — Setup | Repo skeleton, inspect catalog/trace data | ✅ done |
| 1 — Data | `catalog.json` (Individual Test Solutions only) + BM25 index | catalog cleaned; index not built yet |
| 2 — Agent core | Guard layer, slot extractor, policy, retriever, composer (unit tested) | not started |
| 3 — API | `/health`, `/chat`, exact response schema, error handling | not started |
| 4 — Evaluation | Harness over the 10 traces → Recall@10, hard-eval checks, behavior probes | not started |
| 5 — Deploy | Dockerfile, deploy, verify cold start + live endpoints | not started |
| 6 — Docs | 2-page approach document, AI-tool usage disclosure | not started |

### Phase 0 findings

The catalog and trace data were supplied directly (no live scraping needed):

- **`data/raw/catalog_raw.json`** → 377 scraped items, fields: `entity_id, name, link, scraped_at,
  job_levels, job_levels_raw, languages, languages_raw, duration, duration_raw, status, remote,
  adaptive, description, keys`.
- **7 items were Pre-packaged Job Solutions** mixed into the raw scrape (e.g. "Entry Level Cashier
  Solution", "Customer Service Phone Solution" — bundled multi-test products, not tagged separately
  but identifiable by their "X Solution" naming + bundle-description language). These are **out of
  scope** per the brief and are filtered out by `scripts/build_catalog.py`.
- **No explicit `test_type` field exists in the raw data** — SHL's letter codes (A/B/C/D/E/K/P/S) are
  derived from the `keys` category list (`Ability & Aptitude→A, Biodata & Situational Judgment→B,
  Competencies→C, Development & 360→D, Assessment Exercises→E, Knowledge & Skills→K,
  Personality & Behavior→P, Simulations→S`), comma-joined for multi-category items (e.g. `"C, K"`).
  This matches the format used in the provided trace files exactly.
- **`data/catalog.json`** → 370 cleaned Individual Test Solutions, regenerable via
  `python scripts/build_catalog.py`.
- **`eval/traces/C1.md`–`C10.md`** → the 10 provided conversation traces, each a full multi-turn
  transcript with a final labeled shortlist (table on the turn where `end_of_conversation: true`).
  Key behavioral patterns observed, which directly inform the policy design in Phase 2:
  - `recommendations` is empty/null on clarifying turns *and* on refusal turns (e.g. C7 turn 3, a
    legal/HIPAA question) — it is not "last known shortlist," it reflects only what's relevant
    **this turn**.
  - Refinement turns ("keep the shortlist as-is") re-emit the same table rather than omitting it.
  - Comparison questions (C5 turn 2: "difference between OPQ and OPQ MQ Sales Report") get a grounded
    textual answer *and* the shortlist table continues alongside it.
  - No off-topic/prompt-injection example exists in the public 10 traces — guard-layer behavior must
    be built defensively from the spec, not learned from examples (likely covered by holdout probes).

---

## 6. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-01 | LLM provider: **Groq (Llama 3.x)** | Free tier + low latency, fits 30s/call timeout |
| 2026-07-01 | Hosting: **deferred** | Build Dockerized/host-agnostic; pick Render/Fly/HF once app runs locally |
| 2026-07-01 | Retrieval: **BM25 + metadata filters** as the default, embeddings only if eval recall is insufficient | Catalog is small/structured; avoids model-download cold-start cost |
| 2026-07-01 | Exclude 7 "X Solution"-named items from the raw catalog | They are Pre-packaged Job Solutions (multi-test bundles), out of scope per the brief |
| 2026-07-01 | `test_type` derived from the `keys` category list via SHL's A/B/C/D/E/K/P/S legend, comma-joined for multi-category items | No explicit code field in the raw scrape; this format matches the provided trace files exactly |

---

## 7. Open items / blockers

- [x] Catalog data — supplied directly as `catalog.Json`, cleaned into `data/catalog.json` (370 items).
- [x] The 10 public conversation traces — supplied directly, relocated to `eval/traces/`.
- [x] `GROQ_API_KEY` for inference — in local `.env` (gitignored, not committed).
- [ ] Final hosting target for the public endpoint (deferred — Dockerized to stay host-agnostic).

---

## 8. Repo layout (planned)

```
SHL/
├── README.md                  # this file
├── requirements.txt
├── .env                        # GROQ_API_KEY (gitignored, not committed)
├── .gitignore
├── docs/
│   └── approach.md             # final 2-page approach document for submission
├── scripts/
│   └── build_catalog.py        # raw scrape -> data/catalog.json (filter, derive test_type)
├── data/
│   ├── raw/
│   │   └── catalog_raw.json    # original scraped catalog (377 items, as supplied)
│   └── catalog.json            # cleaned Individual Test Solutions catalog (370 items)
├── app/
│   ├── main.py                  # FastAPI app: /health, /chat
│   ├── schemas.py                # Pydantic request/response models
│   ├── guard.py                  # off-topic/injection refusal logic
│   ├── slots.py                   # conversation-state extraction from full history
│   ├── policy.py                   # clarify/recommend/refine/compare decision logic
│   ├── retriever.py                 # BM25 + metadata filtering over catalog.json
│   └── compose.py                    # grounded reply generation
├── eval/
│   ├── traces/                  # C1.md-C10.md, the 10 provided conversation traces
│   └── run_eval.py               # replay harness: Recall@10, hard-evals, behavior probes
├── tests/                       # pytest unit tests per module
└── Dockerfile
```
