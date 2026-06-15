# SNDK Detector Agent

A personal, local-first **research triage engine** for spinoff/carve-out
situations. It ingests candidates in parallel, extracts *objective evidence* from
SEC filings (with source snippets + confidence), scores only what is measurable on
a **100-point weighted model**, hard-fails on survivability/dilution/governance
risk, and has the LLM write a bull case **and** a kill case strictly over that
evidence — then routes each name into reject / watchlist / deep-dive / starter.

The design principle: **the LLM never judges truth.** It classifies events,
extracts claims (quoting the exact filing sentence), and writes memos. All numeric
scoring is deterministic Python. A compelling narrative cannot buy points it can't
back with data. Telegram alerts fire only for the top two tiers; everything else
lives in files you read on your own schedule.

> **v2 vertical slice:** this is built end-to-end for **one event family —
> spinoffs/carve-outs**. Other families (activist, recap, etc.) are a later phase.

```
        ┌──────────────┐
        │  dispatcher  │  (stamps run timestamp)
        └──────┬───────┘
               │  Send fan-out (parallel)
   ┌───────────┼─────────────┐
   ▼           ▼             ▼
ingest_sec  ingest_news  ingest_screener
   └───────────┼─────────────┘
               ▼  (converge → linear chain)
   normalize        entity resolution + persist source_documents
   classify_event   LLM: tag event_family, DROP non-spinoffs
   extract_evidence hybrid yfinance → LLM filing fallback (financials + moat)
   hard_fail        deterministic + LLM disqualifier gate
   score            deterministic 100-pt weighted scorecard
   synthesize       LLM memo (only for ≥ watchlist survivors)
   critic           LLM adversarial pass (break the thesis)
   decide           route to reject / watchlist / deep_dive / starter
   output           write deep-dive memos + ranked watchlist to disk
   alert            Telegram for deep_dive / starter only
```

## The 100-point scorecard

Scored **deterministically** from extracted evidence (weights in parentheses):

1. **Event quality (20)** — confirmed spinoff facts: record date, management
   rationale, standalone financials for the spun entity.
2. **Cycle position (15)** — price near its multi-year low (trough proxy).
3. **Secular tailwind (20)** — a *named* must-have exposure (AI infra, nuclear/SMR,
   defense, quantum, semis…). No named theme ⇒ 0; no generic "growth" credit.
4. **Moat proxies (15)** — sole-source, switching costs, qualification cycles,
   regulatory/capital barriers — each backed by a filing snippet.
5. **Valuation dislocation (20)** — peer-relative multiple gap, computed
   mechanically. **Zero when peers/multiples are unavailable — never fabricated.**
6. **Survivability (15)** — net cash, no near-term maturity wall, positive FCF.

**Hard fails** (refinancing < 12mo, material dilution, customer concentration
without contractual durability, governance red flags, no catalyst path) force
`reject` regardless of score. *"Domain edge"* is removed from the score and kept
as a `priority_for_me` overlay for ranking your own attention.

Tiers (config-driven, 0–100): `reject < TIER_WATCHLIST ≤ watchlist <
TIER_DEEP_DIVE ≤ deep_dive < TIER_STARTER ≤ starter`.

## Tech stack

- **Python 3.11+**, **LangGraph** for orchestration (parallel fan-out via `Send`,
  then a single-writer linear evidence chain)
- **Pydantic** for validating every LLM extraction into an auditable evidence
  contract (`value`, `confidence`, `source_id`, `snippet`)
- **SQLite** (stdlib `sqlite3`, no ORM): candidates, source_documents, evidence,
  scorecards, memos, alerts — keyed on a stable `candidate_id`
- **yfinance** for fundamentals (LLM filing extraction as fallback for fresh spins)
- **OpenAI API** (`OPENAI_MODEL`, default `gpt-4o-mini`)
- **Telegram Bot API** for alerts · **uv** for dependency management

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

Each run prints a summary: candidates per source, spinoffs evaluated, tier
counts, hard-fail count, alerts sent, and every error encountered (nodes never
crash the run — failures are collected and reported). Deep-dive memos and a
ranked watchlist are written to `OUTPUT_DIR` (default `./output`).

## Project layout

```
sndk_detector/
├── state.py              # TypedDicts; candidates/errors use operator.add reducers
├── config.py             # .env loading + loud validation (tier thresholds, etc.)
├── schemas.py            # Pydantic evidence contracts (validate LLM output)
├── scoring.py            # deterministic hard-fail gate + 100-pt weighted score
├── valuation.py          # peer-relative multiple gap + hand-curated PEER_MAP
├── db.py                 # SQLite: candidates, source_documents, evidence,
│                         #         scorecards, memos, alerts (idempotent)
├── graph.py              # LangGraph wiring: parallel ingest → linear evidence chain
├── main.py               # CLI (--once / --schedule), tiered run summary
├── nodes/
│   ├── _llm.py           # shared client, prompt loader, retry, snippet guard
│   ├── ingest_sec.py     # SEC EDGAR FTS (spinoff queries) + source_documents
│   ├── ingest_news.py    # Finnhub news (optional FINNHUB_API_KEY)
│   ├── ingest_screener.py# STUB — India screener (future)
│   ├── normalize.py      # entity resolution, dedupe, persist source_documents
│   ├── classify_event.py # LLM: tag event_family, drop non-spinoffs
│   ├── extract_evidence.py# hybrid yfinance → LLM filing fallback + moat
│   ├── hard_fail.py      # deterministic + LLM disqualifier gate
│   ├── score.py          # valuation gap + weighted scorecard
│   ├── synthesize.py     # LLM memo (≥ watchlist only)
│   ├── critic.py         # LLM adversarial pass
│   ├── decide.py         # tier routing
│   ├── output.py         # write memos + watchlist to disk
│   └── alert.py          # Telegram for deep_dive/starter + mark_as_alerted
└── prompts/
    ├── event_classifier.txt   risk_disqualifier.txt
    ├── financial_extractor.txt memo_writer.txt
    └── moat_extractor.txt      critic.txt
tests/                    # db + migration, scoring, valuation, snippet-guard, SEC parser
```

## Design notes

- **Parallel fan-out, then a single-writer chain.** `graph.py` fans out the
  ingestion nodes via LangGraph's `Send` API (concurrent in one superstep;
  `candidates`/`errors` use `operator.add` reducers). After convergence the
  evidence layers run as a linear chain — single-writer state, no reducers —
  each fanning out per-candidate with an `asyncio` semaphore internally.
- **Evidence over opinion.** Every LLM extraction is validated into a Pydantic
  contract carrying `{value, confidence, source_id, snippet}`. A snippet that
  isn't a verbatim substring of the cited filing is demoted to confidence 0 (a
  cheap, deterministic anti-hallucination guard). Numeric scoring reads only this
  structured evidence — the LLM never sets a number.
- **Token discipline.** Dedup against the DB (`get_recent_scored_ids`) runs
  *before* extraction; `classify_event` drops non-spinoffs before any expensive
  call; the memo + critic are gated on ≥ watchlist; concurrency is bounded and
  rate-limits retry with backoff. More calls per *survivor*, far fewer survivors.
- **Idempotency.** Stable `candidate_id` keys all upserts, dedup, and
  alert-suppression; a failed Telegram send is *not* marked alerted, so it retries
  next run without double-alerting.
- **Defensive everywhere.** Every network/LLM/DB call is wrapped in try/except;
  failures append to `errors` and never crash the graph.

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
The evidence/scoring layers downstream are tested independently (with injected
data) so the pipeline is verifiable even where SEC egress is blocked.

## Extending

- **Generalize beyond spinoffs:** the slice is gated in `classify_event.py`
  (`_KEPT_FAMILIES`) and `ingest_sec.py` (`SEC_QUERIES`). Add an event family by
  widening those, adding family-specific scoring in `scoring.py`, and (if needed)
  a classifier branch — the rest of the chain is family-agnostic.
- **Add an ingestion source:** implement
  `fetch_*_candidates(config) -> (list[Candidate], list[str])` + a
  `make_*_node(config)` factory, then add the node name to `INGESTION_NODES` in
  `graph.py`; it joins the parallel fan-out automatically.
- **Improve fundamentals:** `extract_evidence.py` tries yfinance then falls back
  to LLM filing extraction. Swap in a fundamentals API by adding a provenance
  source there; the scorer already keys off confidence + provenance.

## Tests

```bash
uv run --with pytest python -m pytest -q   # scoring, valuation, snippet guard, db migration
```
