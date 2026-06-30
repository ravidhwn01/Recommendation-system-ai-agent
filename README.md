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
| 1 — Data | `catalog.json` (Individual Test Solutions only) + BM25 index | ✅ done |
| 2 — Agent core | Guard layer, slot extractor, policy, retriever, composer (unit tested) | ✅ done |
| 3 — API | `/health`, `/chat`, exact response schema, error handling | ✅ done |
| 4 — Evaluation | Harness over the 10 traces → Recall@10, hard-eval checks, behavior probes | ✅ done (re-run pending fresh quota) |
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

### Phase 1 findings

- `app/retriever.py`: BM25 index (`rank_bm25`) over `name`(×4) + `categories`(×2) + `description`(×1)
  + `job_levels`(×1) tokens — repetition is the standard field-boosting trick since `rank_bm25` has no
  native per-field weights. Hard metadata filters (`test_type`, `job_levels`, `max_duration_minutes`,
  `remote_only`, `language`) are applied **before** ranking, so a filtered-out item can never be
  crowded out of relevance by an irrelevant high scorer — and never sneak past a filter either.
  `find_by_name()` does fuzzy product-name lookup for grounding comparison questions.
- Smoke-tested against trace-style queries (`tests/test_retriever.py`, 10/10 passing): exact product
  names ("OPQ32r") and sharp domain terms ("HIPAA") rank correctly at #1. Broader multi-concept
  queries (e.g. "Rust engineer high performance networking") only recover part of a trace's full
  expected shortlist in the raw top-5 — expected, since a single BM25 call isn't meant to be the final
  answer. Phase 2's policy will pull a wider candidate pool (top_k ≈ 15–20), possibly via multiple
  sub-queries per extracted slot, and let the LLM select/rerank within that pool.

### Phase 2 findings

Implemented as a compound pipeline (`app/pipeline.py`): `guard` -> `slots` (interpret) -> `policy`
(decide) -> `retriever` (only when warranted) -> `compose`. At most **2 LLM calls per turn** -
refusals and clarifying questions never call the LLM at all (deterministic templates / passthrough
of the interpret call's own suggested question), which keeps latency low and refusal behavior
un-overridable by clever phrasing.

- `app/guard.py` - regex-based prompt-injection detection that runs **before** any LLM call, on the
  raw last user message. This is deliberate: an injection's whole point is to make an LLM disobey its
  instructions, so the first line of defense can't itself depend on an LLM being faithfully obedient.
  Nuanced off-topic/legal-advice judgment is left to the LLM interpret call instead, since heuristics
  there would be too prone to false positives (e.g. "is this EEOC compliant" is in-scope-adjacent).
- `app/slots.py` - single JSON-mode LLM call that reads the **entire** history every time (stateless
  by construction) and outputs scope, intent, accumulated constraints, and turn-state signals
  (`ready_to_recommend`, `prior_recommendations_given`, `user_is_closing`). Offloading state
  reconstruction to the LLM (rather than Python-side diffing) is what makes refinement
  ("actually add a cognitive ability test") and re-confirmation work correctly without server-side
  session state.
- `app/policy.py` - pure Python decision logic on top of the interpretation. Key rule found by
  testing: **recommend-readiness and comparison-readiness are independent flags**
  (`should_recommend` vs `comparison_requested`). Initially comparison questions were forced down the
  same path as recommendations, which caused a context-free "what's the difference between X and Y"
  question to spuriously attach an unrelated shortlist (caught via smoke test - see below). Decoupling
  them means a standalone comparison gets a grounded text answer with `recommendations: []`, while a
  comparison asked *after* a shortlist already exists (matching trace C5) correctly continues
  showing/refreshing that shortlist. Turn-budget is enforced here too: clarifying is only allowed for
  the first two exchanges (`len(messages) < 5`, matching the pattern observed in trace C1), and a
  `force_commit` flag instructs the composer it **must** select at least one candidate rather than
  stalling once the budget is tight.
- `app/compose.py` - the only LLM call that can produce `recommendations`. The model selects indices
  into an already-retrieved candidate list and writes the reply; Python then maps indices back to
  catalog rows for the actual `name`/`url`/`test_type`, so a hallucinated item is structurally
  impossible (the model can't introduce a name/URL that isn't in the list it was given). On LLM
  failure, falls back to top-ranked BM25 candidates with templated text rather than erroring.
- `eval/smoke_test.py` - manual integration script (real Groq calls, not part of default `pytest`
  since it needs network + API key) exercising clarify / recommend / refine / compare / legal-refusal
  / off-topic-refusal / injection / force-commit. This is what caught the comparison-bleed bug above
  before it reached Phase 4 evaluation - worth running again after any prompt or policy change.
- `tests/test_guard.py`, `tests/test_policy.py` - 13 new unit tests, all deterministic (no LLM calls,
  no network), covering injection detection and every policy branch including turn-cap edges. Full
  suite: 23/23 passing.

### Phase 3 findings

- `app/main.py` - thin FastAPI wrapper: `GET /health` returns `{"status": "ok"}`; `POST /chat` runs
  `pipeline.handle_chat` (synchronous - uses the sync Groq client) in a threadpool via
  `run_in_threadpool`, wrapped in `asyncio.wait_for` with a 25s hard deadline (`CHAT_HARD_DEADLINE_SECONDS`
  in `app/config.py`). If that deadline trips, or `handle_chat` raises for any unanticipated reason, the
  endpoint returns a templated fallback `ChatResponse` (HTTP 200, valid schema, empty recommendations)
  rather than a 500 or a hang - matching the "graceful degradation" guardrail rather than letting a
  single bad turn fail the whole evaluator run.
- The BM25 index is built once at FastAPI startup (`lifespan` calls `get_index()`), not lazily on the
  first request, so the first real `/chat` call after a cold start isn't doing index-build work on top
  of LLM latency.
- Tightened the LLM timeout budget after thinking through worst-case latency: Groq's SDK retries
  transient errors by default, which could silently multiply a single call's latency well past its
  stated timeout. Set `max_retries=0` on the Groq client and dropped `LLM_TIMEOUT_SECONDS` to 8s, so
  two LLM calls in the worst case cost at most 16s - comfortably inside both the 25s app-level deadline
  and the evaluator's 30s/call cap.
- `tests/test_api.py` - 6 tests using `TestClient` with `handle_chat` mocked (fast, deterministic, no
  LLM calls): exact response-shape compliance, empty-recommendations serialization, graceful fallback
  on a simulated pipeline exception, and 422 rejection of malformed requests (bad `role`, missing
  `messages`). Full suite: 29/29 passing.
- Verified manually over real HTTP (not just `TestClient`) by running `uvicorn app.main:app` and
  curling `/health` and `/chat` directly - confirmed a vague query correctly returns an empty-shortlist
  clarifying response, a comparison question returns a grounded answer with `recommendations: []`, and
  a malformed request returns 422, all through the actual ASGI server rather than the in-process test
  client.
- Live demo run through the actual server surfaced a real retrieval bug: replaying the spec's own
  example conversation (Java developer -> mid-level -> "actually, add personality tests") returned the
  same 3 Java/K-type items unchanged - the refinement was silently ignored. Root cause: a single BM25
  query text biased toward "Java developer" terms ranks K-type items so far above P-type items that
  none made the top-20 candidate pool at all, even though the `test_type` filter correctly allowed both.
  Fixed in `app/pipeline.py::_retrieve_candidates`: when more than one test_type is requested, retrieve
  per type and merge the results, so every requested category gets guaranteed representation regardless
  of how the free-text query is worded. Re-verified live: the same conversation now correctly returns
  the 3 Java tests plus 2 personality/competency items. Added `tests/test_pipeline_retrieval.py` as a
  regression test. Full suite: 32/32 passing.

### Phase 4 findings

- `eval/parse_traces.py` - parses each `eval/traces/C*.md` file into the scripted sequence of user
  messages plus the labeled expected shortlist (the table on the turn marked `end_of_conversation: true`).
  This is a **scripted replay** harness, not a full user-simulation harness: the real evaluator drives an
  LLM-simulated user persona that adapts to our agent's actual replies, whereas this replays the trace's
  exact recorded user messages regardless of what our agent says back. That's a fast, deterministic,
  zero-extra-cost proxy for local tuning, but it can diverge from the real evaluator's grading - e.g. if
  our agent asks a different clarifying question than the trace's recorded assistant turn, replaying the
  trace's next scripted user line may answer a question we never actually asked.
- `eval/run_eval.py` - replays all 10 traces directly against `pipeline.handle_chat` (no HTTP, for speed),
  computes Recall@10 by URL match against each trace's expected shortlist, validates hard-eval constraints
  (schema, catalog-only URLs, turn cap) across **every** turn of every trace (not just the final one), and
  runs 6 behavior probes (off-topic refusal, no-recommend-on-vague-turn-1, legal-advice refusal,
  prompt-injection refusal, refinement honored, comparison grounded without a spurious shortlist).

**Rate limits dominated this phase and are themselves a real finding.** Running 10 traces back-to-back
(~45 turns, up to 2 LLM calls each) repeatedly tripped Groq free-tier limits: first the per-minute cap
(12,000 TPM on `llama-3.3-70b-versatile`), then - after enough cumulative testing in one day across
multiple accounts - the **daily cap (100,000 TPD)**, which is per-organization and does not reset by
generating a new API key within the same account. Mitigations applied directly to `app/llm.py` (not just
the eval script, since this is a real production risk too): `max_retries=0` on the Groq client with one
deliberate, capped backoff-and-retry specifically for `RateLimitError` (other failures still fail fast to
protect the `/chat` hard deadline), plus trimming `CANDIDATE_POOL_SIZE` and per-candidate description
length in `app/compose.py` to cut prompt size. None of this eliminates the daily cap, only the per-minute
one. **Production implication**: the real evaluator's traffic is one paced conversation at a time, not a
tight batch loop, so it's far less likely to trip the per-minute limit than this harness - but a Dev-tier
or alternate-provider key is worth having before relying on heavy iteration close to submission time.

Across multiple runs, mean Recall@10 ranged from 0.02 (daily-quota-exhausted runs, where nearly every
LLM call failed and fell back to generic templated behavior) to 0.321 (a run that hit only per-minute
limits, with most calls eventually succeeding after backoff). The 0.321 run is the most trustworthy
number obtained so far; **a clean, fully-successful run is still pending a fresh-quota key** and should be
re-run before final submission. Two findings survived even the worst, near-total-LLM-failure runs:
**hard-evals passed with 0 violations in every single run** (the Python-side catalog-grounding and schema
guarantees hold regardless of LLM call success - the whole point of building them that way), and the
off-topic/legal/injection refusal probes plus the comparison-grounding probe passed consistently (5/6).

**A real, reproducible recall gap, not noise**: "Occupational Personality Questionnaire OPQ32r" was
missed across most traces in every run, including the cleaner one. Cross-referencing all 10 traces showed
it isn't random - OPQ32r (SHL's flagship, broadest-coverage personality instrument) recurs as a default
complement to a primarily technical/skill-focused shortlist in 6 of 10 expected answers, several with no
personality test_type ever explicitly requested by the user (e.g. a pure "Java, Spring, SQL, AWS, Docker"
request still expected OPQ32r). Root cause: when `test_types` is inferred as e.g. `{'K'}` from a technical
request, the main retrieval never even fetches a personality candidate for compose to consider, and a
generic BM25 search among personality items has no tech-specific terms to rank OPQ32r above any other
personality item anyway. Fixed in `app/pipeline.py::_supplement_with_personality`: whenever a shortlist
is being built and personality wasn't explicitly excluded, look up OPQ32r by name and add it to the
candidate pool (respecting any other active filters like duration/remote/language), in addition to a
small generic personality-type search. Verified directly (not via noisy full-harness runs) via
`tests/test_pipeline_retrieval.py` and isolated single-trace replays - confirmed OPQ32r now reaches the
candidate pool and is selected in cases it previously wasn't (e.g. C2's recall went 0.2 -> 0.4 in an
isolated before/after test). This is intentionally **not** a forced inclusion - `compose.py`'s LLM still
decides whether to select it, so traces where OPQ32r genuinely isn't expected (C3, C6, C10) aren't
penalized by always force-adding it.

**A second bug found through this same investigation**: the personality supplement was originally
*appended* to the end of the candidate list. `compose.py`'s graceful-degradation fallback (when the LLM
call itself fails) takes `candidates[:5]` - which silently sliced off the supplement every time, i.e.
exactly when rate-limit pressure was highest and the supplement mattered most. Fixed by prepending instead
of appending. Added two regression tests in `tests/test_pipeline_retrieval.py` that verify this without
any live API call (one checks candidate ordering directly, one mocks the LLM call to fail and asserts
OPQ32r still survives the fallback path) - full suite now 34/34 passing.

**One probe remains a genuine, reproducible fail**: `honors_refinement_edit` failed in every run so far.
Given the rate-limit pressure during every run to date, this is plausibly explained by the interpret call
itself failing on the refinement turn and falling back to empty constraints (losing the "Java developer"
context entirely, not just the personality addition) - but this hasn't been confirmed under clean
conditions yet and should be re-checked on the pending clean re-run rather than assumed fixed.

---

## 6. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-01 | LLM provider: **Groq (Llama 3.x)** | Free tier + low latency, fits 30s/call timeout |
| 2026-07-01 | Hosting: **deferred** | Build Dockerized/host-agnostic; pick Render/Fly/HF once app runs locally |
| 2026-07-01 | Retrieval: **BM25 + metadata filters** as the default, embeddings only if eval recall is insufficient | Catalog is small/structured; avoids model-download cold-start cost |
| 2026-07-01 | Exclude 7 "X Solution"-named items from the raw catalog | They are Pre-packaged Job Solutions (multi-test bundles), out of scope per the brief |
| 2026-07-01 | `test_type` derived from the `keys` category list via SHL's A/B/C/D/E/K/P/S legend, comma-joined for multi-category items | No explicit code field in the raw scrape; this format matches the provided trace files exactly |
| 2026-07-01 | At most 2 LLM calls per turn (interpret + compose); refusals/clarifications use 0 LLM calls | Fits the 30s/call budget with headroom; refusal text being templated (not generated) means it can't be talked out of refusing |
| 2026-07-01 | `should_recommend` and `comparison_requested` are independent policy flags, not one combined branch | A context-free comparison question was spuriously attaching an unrelated shortlist when comparison forced the same path as recommend-readiness; found via `eval/smoke_test.py` |
| 2026-07-01 | Recommendations are built by mapping LLM-selected indices back to catalog rows in Python, never from LLM-generated text | Structurally prevents hallucinated names/URLs from ever reaching the response |
| 2026-07-01 | `app/llm.py` does one capped retry-with-backoff specifically for `RateLimitError`, other failures still fail fast | Found Groq free-tier TPM/TPD limits get tripped by realistic conversation volume; a 429 is recoverable and worth one short wait, unlike other failure modes where retrying risks blowing the `/chat` deadline |
| 2026-07-01 | OPQ32r is looked up by name and prioritized in the candidate pool whenever a shortlist is built and personality isn't explicitly excluded, rather than relying on generic P-type BM25 ranking | Empirically the default complement to technical shortlists in 6/10 traces, consistent with it being SHL's flagship personality instrument in real practice - not just trace-overfitting | 
| 2026-07-01 | Eval harness is scripted replay (exact trace user messages), not LLM-simulated user | Fast, deterministic, zero extra API cost for local tuning; documented as an approximation of the real evaluator, not equivalent to it |

---

## 7. Open items / blockers

- [x] Catalog data — supplied directly as `catalog.Json`, cleaned into `data/catalog.json` (370 items).
- [x] The 10 public conversation traces — supplied directly, relocated to `eval/traces/`.
- [x] `GROQ_API_KEY` for inference — in local `.env` (gitignored, not committed).
- [ ] Final hosting target for the public endpoint (deferred — Dockerized to stay host-agnostic).
- [ ] **A clean, fully-successful eval run** — every run to date hit Groq free-tier rate limits (per-minute
      and, after enough same-day testing, per-organization daily caps), so the Recall@10 numbers in
      `eval/results.json` are not yet trustworthy as a final figure. Needs a fresh-quota key (genuinely
      different account, not just a new key in an already-used account) or a paid tier, ideally close to
      final submission so the number reflects the final code.
- [ ] Re-check the `honors_refinement_edit` probe under clean (non-rate-limited) conditions — failed in
      every run so far, plausibly explained by interpret-call failures under rate-limit pressure on that
      specific turn, but not yet confirmed as fixed vs. a real remaining bug.

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
│   ├── config.py                 # env loading: GROQ_API_KEY/MODEL, timeouts, turn-budget constants
│   ├── schemas.py                 # Pydantic request/response models
│   ├── llm.py                      # Groq client wrapper (chat_json / chat_text, fails soft)
│   ├── guard.py                     # heuristic prompt-injection detection (pre-LLM)
│   ├── slots.py                      # LLM interpret call: scope, intent, constraints, turn-state
│   ├── policy.py                      # deterministic turn-budget-aware action decision
│   ├── retriever.py                    # BM25 + metadata filtering over catalog.json
│   ├── compose.py                       # refusal templates, clarify passthrough, recommend LLM call
│   └── pipeline.py                       # orchestrates guard->slots->policy->retriever->compose
├── eval/
│   ├── traces/                  # C1.md-C10.md, the 10 provided conversation traces
│   ├── smoke_test.py             # manual integration check against live Groq (not in pytest)
│   ├── parse_traces.py            # trace .md -> structured TraceCase (scripted user turns + expected shortlist)
│   ├── run_eval.py                 # replay harness: Recall@10, hard-evals, behavior probes
│   └── results.json                 # latest run's output (caveated - see "Phase 4 findings")
├── tests/                       # pytest unit tests per module (guard, policy, retriever, pipeline, api)
└── Dockerfile
```
