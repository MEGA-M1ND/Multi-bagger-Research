# SNDK Detector Agent

A personal, local-first stock-screening agent. It ingests candidates from
multiple sources in parallel, grounds each with real financial data + cited
research, scores them against a fixed **6-point blueprint** using an LLM, and
sends a Telegram alert for high-scoring matches. Single-user, runs on a
schedule, built for correctness and observability over scale.

```
        ┌──────────────┐
        │  dispatcher  │  (stamps run timestamp)
        └──────┬───────┘
               │  Send fan-out (parallel)
   ┌───────────┼─────────────┐
   ▼           ▼             ▼
ingest_sec  ingest_news  ingest_screener
   └───────────┼─────────────┘
               ▼  (converge)
            ┌────────┐
            │ enrich │  Perplexity: structured financials + cited research
            └───┬────┘  (no-op without PERPLEXITY_API_KEY)
                ▼
            ┌───────┐
            │ score │  async, semaphore-bounded LLM scoring
            └───┬───┘
                ▼
            ┌───────┐
            │ alert │  filter ≥ threshold & not-yet-alerted → Telegram
            └───────┘
```

## The 6-point blueprint

Each candidate is scored TRUE/FALSE on six factors (total 0–6):

1. **Structural event** — spinoff, carve-out, IPO, restructuring, or management
   change that could re-rate the company.
2. **Cyclical trough** — at/near a multi-year low in margins/pricing/sentiment
   with suppressed earnings that could recover sharply.
3. **Secular tailwind** — a direct bottleneck or must-have supplier in AI,
   robotics, nuclear/SMR, defense AI, or quantum (not peripheral).
4. **Supply constraint** — a moat competitors can't replicate in < 2 years
   (patents, capital intensity, regulatory moats, rare materials, talent).
5. **Undervalued narrative** — valuation doesn't yet reflect the growth story;
   trades like a legacy/cyclical business.
6. **Domain edge** — someone with AI infra/security expertise would understand
   the technical moat better than the average investor.

A candidate that scores **≥ `BLUEPRINT_THRESHOLD`** (default 4) gets an
LLM-generated thesis and a Telegram alert.

## Tech stack

- **Python 3.11+**, **LangGraph** for orchestration (parallel fan-out via `Send`)
- **SQLite** (stdlib `sqlite3`, no ORM) for persistence + idempotency
- **OpenAI API** for scoring (`OPENAI_MODEL`, default `gpt-4o-mini`)
- **Perplexity API** (optional) for enrichment — structured financials + cited
  research (`PERPLEXITY_MODEL`, default `sonar`); OpenAI-compatible, so it reuses
  the same SDK
- **Telegram Bot API** for alerts
- **uv** for dependency management

## Setup

```bash
# 1. Install dependencies into a managed venv
uv sync

# 2. Configure
cp .env.example .env
# edit .env: OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID are required.
# IMPORTANT: set a real SEC_USER_AGENT with your contact info — SEC returns 403
# without a descriptive User-Agent.
```

Required keys (validated loudly at startup): `OPENAI_API_KEY`,
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Everything else has sane defaults — see
`.env.example`.

## Usage

```bash
# Single run
uv run python -m sndk_detector.main --once

# Scheduled loop (every 6h by default; no cron dependency)
uv run python -m sndk_detector.main --schedule
uv run python -m sndk_detector.main --schedule --interval 3   # every 3h

# Verbose logging
uv run python -m sndk_detector.main --once -v
```

If installed as a script (`uv pip install -e .`), the `sndk-detector` command is
also available.

Each run prints a summary: candidates per source, how many were scored, how many
alerted, and every error encountered (sources never crash the run — failures are
collected and reported).

## Project layout

```
sndk_detector/
├── state.py              # TypedDicts + reducers (Annotated[..., operator.add])
├── config.py             # .env loading + loud validation
├── db.py                 # SQLite: candidates + alerts, idempotent by candidate_id
├── graph.py              # LangGraph wiring + Send-based parallel fan-out
├── main.py               # CLI (--once / --schedule), run summary
├── nodes/
│   ├── ingest_sec.py     # SEC EDGAR full-text search (fully implemented)
│   ├── ingest_news.py    # STUB — clean interface, fill in your news source
│   ├── ingest_screener.py# STUB — clean interface, fill in your India screener
│   ├── enrich.py         # Perplexity grounding: financials + cited research
│   ├── score_blueprint.py# async scoring, semaphore + retry/backoff, dedup
│   └── alert.py          # filter + Telegram send + mark_as_alerted
└── prompts/
    ├── blueprint_scorer.txt   # 6-factor criteria, JSON output
    ├── enrich_financials.txt  # structured-financials extraction (JSON)
    ├── enrich_research.txt     # grounded qualitative research
    └── thesis_generator.txt   # thesis for candidates over threshold
tests/                    # db, state, SEC-parser, and enrich-parser tests
```

## Design notes

- **Parallel fan-out.** `graph.py` uses LangGraph's `Send` API: the dispatcher's
  conditional edge returns a *list* of `Send` objects (one per source) so all
  ingestion runs concurrently in one superstep, then converges on scoring. The
  `candidates`/`errors` state fields use `operator.add` reducers so parallel
  branches concatenate instead of overwriting.
- **Token discipline.** Scoring dedups against the DB first (skips anything
  scored within `DEDUP_LOOKBACK_DAYS`), bounds concurrency with an `asyncio`
  semaphore (`MAX_CONCURRENT_LLM`), retries rate-limits with exponential backoff,
  and only generates a (second, costlier) thesis call for candidates over the
  threshold.
- **Enrichment before scoring.** Four of the six factors need *current,
  company-specific facts* the model can't reliably know. The `enrich` node
  grounds each post-dedup candidate with Perplexity: a structured-JSON financials
  call (price, market cap, margins, valuation multiples, dated corporate events —
  stored as *numbers*, not prose, so the number-driven factors stay deterministic
  and citable) plus a grounded, cited research summary for the qualitative
  factors. The data lands in `raw_data` (which the scorer already serializes), so
  no scorer rewrite is needed. It's a no-op without `PERPLEXITY_API_KEY`, dedups
  on `last_enriched` within `RESEARCH_LOOKBACK_DAYS`, and is capped per run by
  `MAX_ENRICH_PER_RUN`.
- **Idempotency.** Every candidate has a stable `candidate_id` (hash of
  ticker+source). Upserts, dedup, and alert-suppression all key off it; a failed
  Telegram send is *not* marked alerted, so it retries next run without
  double-alerting.
- **Defensive ingestion.** Every network call is wrapped in try/except; failures
  append to `errors` and never crash the graph.

## ⚠️ SEC parser note (read this)

The build spec asked for one live SEC call to verify the JSON shape before
writing the parser. **In the environment where this was built, the SEC hosts
(`efts.sec.gov`, `data.sec.gov`) were blocked by a network egress allowlist**, so
the live call returned `403 — Host not in allowlist`.

The parser in `ingest_sec.py` is therefore written against the **documented,
stable** EDGAR full-text search shape (`hits.hits[]._source` with
`display_names`, `ciks`, `form`, `file_date`, …). It is fully defensive and
includes a shape-guard that surfaces a loud error if the live response doesn't
match. Tests in `tests/test_ingest_sec.py` validate the parser against a
documented-shape fixture.

**When you run this with SEC reachable:** if you see
`unexpected SEC response shape` in the run errors, paste a real sample response
and adjust `_extract_candidate` — it's the only function that needs to change.

## ⚠️ Perplexity enrichment note (read this)

Same story as SEC: `api.perplexity.ai` was blocked by the egress allowlist in the
build environment, so the one live `finance_search` call to verify its response
shape returned `403`. `nodes/enrich.py` is therefore written against Perplexity's
**documented, OpenAI-compatible** `/chat/completions` endpoint — it asks the
model (with its built-in web/finance grounding) to return a JSON object in *our*
schema plus citations. All request shaping and response parsing
(`_parse_financials`, `_apply_financials`, `_extract_citations`) are isolated so
that, once egress is opened and the live `finance_search` structured shape is
confirmed, only those spots need editing. Everything is defensive: missing fields
degrade to absent and never crash the graph.

**To enable it:** add `api.perplexity.ai` to your egress allowlist, set
`PERPLEXITY_API_KEY` in `.env`, then run. India fundamentals coverage is
unverified — missing fields simply don't appear in `raw_data` (the scorer already
tolerates sparse data).

## Extending

- **Add an ingestion source:** copy a stub in `nodes/`, implement
  `fetch_*_candidates(config) -> (list[Candidate], list[str])` and a
  `make_*_node(config)` factory, then add the node name to `INGESTION_NODES` in
  `graph.py`. It's automatically included in the parallel fan-out.
- **Fill in the stubs:** `ingest_news.py` and `ingest_screener.py` have clean
  interfaces and TODO sketches. They currently return no candidates (and an
  honest note) rather than fabricating data.

## Tests

```bash
uv run pytest -q
```
