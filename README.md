# Multi-bagger-Research

# Build: SNDK Detector Agent

## Goal
A personal stock-screening agent that ingests data from multiple sources, scores
candidates against a fixed 6-point "blueprint" via an LLM, and sends alerts for
high-scoring matches. Single-user, runs on a schedule, local-first. Prioritize
correctness and observability over scale.

## Tech stack
- Python 3.11+, LangGraph for orchestration
- SQLite for persistence (via sqlite3 stdlib, no ORM)
- OpenAI API for scoring (model configurable, default gpt-4o-mini for cost)
- Telegram Bot API for alerts
- python-dotenv for config
- Use `uv` for dependency management

## Architecture
Three logical stages: ingestion (fan-out, parallel) → scoring → filter → alert.

Use LangGraph's `Send` API for true parallel fan-out across ingestion sources.
A dispatcher node should fan out to all enabled ingestion nodes simultaneously,
then converge to the scoring node. Do NOT chain ingestion nodes linearly.

## State schema (state.py)
Implement these TypedDicts. Use `Annotated[List[Candidate], operator.add]` for
fields that accumulate across parallel nodes.

- BlueprintScore: six booleans (structural_event, cyclical_trough,
  secular_tailwind, supply_constraint, undervalued_narrative, domain_edge),
  total_score int, reasoning str
- Candidate: ticker, company_name, market, source, raw_data, blueprint (optional),
  thesis (optional), price (optional), market_cap (optional), alerted bool,
  candidate_id (stable hash of ticker+source for dedup)
- AgentState: candidates (accumulate), scored_candidates, alert_queue,
  run_timestamp, errors (accumulate)

## Ingestion nodes (nodes/)
Each must be defensive: wrap network calls in try/except, append failures to
`errors`, never crash the graph. Cap results per source (default 20).

1. ingest_sec.py — SEC EDGAR full-text search. Use the correct endpoint:
   `https://efts.sec.gov/LATEST/search-index?q="spin-off"&forms=8-K,S-1,10-12B`
   Set a real User-Agent header (SEC requires it). VERIFY the response JSON shape
   by making one real call and inspecting it before writing the parser — do not
   assume field names. Respect SEC's 10 req/sec rate limit.
2. ingest_news.py — stub with a clean interface (a function that returns
   list[Candidate]). Leave a TODO for the actual news source. Don't fake data.
3. ingest_screener.py — stub for India stocks, same pattern.

Build SEC fully; news and screener as honest stubs I can fill in.

## Scoring node (nodes/score_blueprint.py)
- Batch concerns: candidates may number 40+. Process with a concurrency limit
  (asyncio + semaphore, max 5 concurrent LLM calls) and add retry-with-backoff
  on rate-limit errors. Do NOT call the LLM sequentially in a plain loop.
- Two prompts loaded from prompts/ as text files: blueprint_scorer and
  thesis_generator. Use response_format json_object for scoring.
- Only generate a thesis when total_score >= BLUEPRINT_THRESHOLD.
- Dedup against the DB before scoring — skip candidates already scored recently
  (configurable lookback, default 7 days) to avoid burning tokens.

## Persistence (db.py)
SQLite with tables: candidates (with scores, JSON blueprint, timestamps) and
alerts (what was sent, when). Functions: init_db, upsert_candidate,
get_recent_candidate_ids, mark_as_alerted, has_been_alerted. Use candidate_id
for idempotency.

## Alert node (nodes/alert.py)
- Filter: only alert candidates that scored >= threshold AND haven't been
  alerted (check DB). 
- Telegram with Markdown. Handle the send failure case gracefully.
- After successful send, mark_as_alerted in DB.

## Config (config.py)
Load from .env: OPENAI_API_KEY, OPENAI_MODEL, TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID, BLUEPRINT_THRESHOLD (default 4), DEDUP_LOOKBACK_DAYS (default 7),
MAX_CANDIDATES_PER_SOURCE (default 20), MAX_CONCURRENT_LLM (default 5).
Validate required keys are present at startup; fail loudly if missing.

## Entry point (main.py)
- CLI with argparse: `--once` (single run) and `--schedule` (loop every N hours,
  default 6, using a simple sleep loop — no cron dependency).
- Print a clear run summary: sources hit, candidates found, scored, alerted, errors.
- init_db on startup.

## Prompts (prompts/)
Create blueprint_scorer.txt and thesis_generator.txt. I'll provide the scoring
criteria below — embed them verbatim into the scorer prompt.

[BLUEPRINT CRITERIA — the 6 factors]
1. STRUCTURAL_EVENT: recent/upcoming spinoff, carve-out, IPO, restructuring, or
   management change that could re-rate the company.
2. CYCLICAL_TROUGH: at/near multi-year low in margins, pricing, or sentiment with
   suppressed earnings that could recover sharply.
3. SECULAR_TAILWIND: a direct bottleneck or critical must-have supplier in AI,
   robotics, nuclear/SMR, defense AI, or quantum — not peripheral.
4. SUPPLY_CONSTRAINT: genuine constraint competitors can't replicate in <2 years
   (patents, capital intensity, regulatory moats, rare materials, specialized talent).
5. UNDERVALUED_NARRATIVE: market valuation doesn't yet reflect the secular growth
   story; trading like a legacy/cyclical business.
6. DOMAIN_EDGE: someone with AI infrastructure/security expertise would understand
   the technical moat better than the average investor.

## Build order
1. Scaffold structure, requirements, .env.example, README
2. state.py, config.py, db.py (with a quick test that init_db works)
3. ingest_sec.py — make ONE real SEC call, inspect the JSON, THEN write the parser
4. score_blueprint.py with the async/concurrency handling
5. alert.py
6. graph.py wiring with Send-based fan-out
7. main.py
8. End-to-end dry run with --once; show me the output

## Constraints
- Don't invent API response shapes — verify with a real call or tell me you can't
  reach it and need me to paste a sample.
- Comment the non-obvious parts (the Send fan-out, the async semaphore).
- Keep it readable. This is a personal tool I'll be extending.
Multi-bagger Research
