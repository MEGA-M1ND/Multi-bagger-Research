"""Enrichment node: ground each candidate with Perplexity before scoring.

Two pieces of value are attached to every candidate the stage can afford:

  1. Structured financials (``finance_search``-grounded): latest price, market
     cap, a small fundamentals block (margins, valuation multiples, trend), and
     dated corporate events. These are stored as *numbers*, not prose, so the
     number-driven blueprint factors (cyclical trough, undervalued narrative)
     stay deterministic and citable.
  2. Grounded qualitative research: a short, cited narrative for the factors
     that need current, company-specific facts (secular tailwind, supply
     constraint, structural event, domain edge).

The enriched data lands in ``candidate["raw_data"]`` (plus ``price`` /
``market_cap``), which ``score_blueprint._candidate_block`` already serializes
into the scorer prompt — so no scorer code change is required.

SHAPE NOTE — IMPORTANT (same honesty as ``ingest_sec.py``):
  ``api.perplexity.ai`` is blocked by this environment's egress allowlist, so the
  one real ``finance_search`` call the plan calls for returns HTTP 403 and could
  not be made. This module is therefore written against Perplexity's DOCUMENTED
  OpenAI-compatible chat endpoint (``/chat/completions``), asking the model — with
  its built-in web/finance grounding — to return a JSON object in *our* schema
  plus citations. All request shaping (the prompts + ``create`` calls) and
  response parsing (``_parse_financials`` / ``_extract_citations``) are isolated
  so that, once egress is opened and the live ``finance_search`` response shape is
  confirmed, the structured tool output can be wired in by editing only those
  spots. Everything is defensive: missing/garbled fields degrade to absent and
  never crash the graph.

In-place mutation:
  The node mutates candidate dicts in place and deliberately does NOT return the
  ``candidates`` channel — that channel uses an additive reducer, so returning it
  would duplicate every candidate. The compiled graph has no checkpointer, so the
  scorer downstream reads the very same dict objects (mutations are visible).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from openai import AsyncOpenAI

from ..config import Config
from ..db import get_recently_enriched_ids, upsert_candidate
from ..state import Candidate

# Reuse the scorer's retry/backoff and prompt loader rather than duplicating them.
from .score_blueprint import _call_with_retry, _load_prompt

logger = logging.getLogger(__name__)

# Perplexity is OpenAI-compatible; point the same SDK at its base URL.
_PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any) -> Optional[float]:
    """Coerce a model-supplied value to float, or None if it isn't a number."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_financials(content: Optional[str]) -> dict:
    """Defensively parse the financials JSON. Returns {} on any problem."""
    try:
        data = json.loads(content or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _extract_citations(resp: Any) -> List[str]:
    """Best-effort pull of source URLs from a Perplexity response.

    Perplexity returns citations as a non-standard top-level field; the key has
    shifted across versions (``citations`` vs ``search_results``), so we probe
    both via the model dump. UNVERIFIED against a live call (egress blocked).
    """
    try:
        data = resp.model_dump()
    except Exception:  # noqa: BLE001 - never let citation parsing crash enrichment
        return []
    cites = data.get("citations")
    if isinstance(cites, list):
        return [str(c) for c in cites if c]
    results = data.get("search_results")
    if isinstance(results, list):
        out: List[str] = []
        for r in results:
            if isinstance(r, dict):
                out.append(str(r.get("url") or r.get("title") or r))
            elif r:
                out.append(str(r))
        return out
    return []


def _apply_financials(candidate: Candidate, data: dict, citations: List[str]) -> None:
    """Map parsed financials JSON onto the candidate (numbers, not prose)."""
    price = _as_float(data.get("price"))
    if price is not None:
        candidate["price"] = price
    market_cap = _as_float(data.get("market_cap"))
    if market_cap is not None:
        candidate["market_cap"] = market_cap

    raw = candidate.setdefault("raw_data", {})
    fundamentals = data.get("fundamentals")
    if isinstance(fundamentals, dict) and fundamentals:
        raw["fundamentals"] = fundamentals
    events = data.get("corporate_events")
    if isinstance(events, list) and events:
        raw["corporate_events"] = [str(e) for e in events if e]
    currency = data.get("currency")
    if currency:
        raw["currency"] = str(currency)
    if citations:
        raw["financial_citations"] = citations


async def _enrich_one(
    client: AsyncOpenAI,
    config: Config,
    semaphore: asyncio.Semaphore,
    candidate: Candidate,
) -> Tuple[Candidate, List[str]]:
    """Enrich a single candidate in place. Returns (candidate, soft_errors).

    The two Perplexity calls fail independently: a failure in one is recorded
    and the other still runs. The candidate is never dropped on failure — it
    simply flows to scoring without that piece of grounding.
    """
    ticker = candidate.get("ticker")
    company = str(candidate.get("company_name") or "")
    market = str(candidate.get("market") or "")
    errors: List[str] = []

    async with semaphore:
        # --- 1. structured financials -------------------------------------
        fin_prompt = (
            _load_prompt("enrich_financials")
            .replace("{ticker}", str(ticker or ""))
            .replace("{company_name}", company)
            .replace("{market}", market)
        )
        try:
            resp = await _call_with_retry(
                lambda: client.chat.completions.create(
                    model=config.perplexity_model,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    messages=[{"role": "user", "content": fin_prompt}],
                ),
                what=f"enrich.fin[{ticker}]",
            )
            data = _parse_financials(resp.choices[0].message.content)
            if data:
                _apply_financials(candidate, data, _extract_citations(resp))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the graph
            errors.append(f"enrich: {ticker} financials failed: {exc}")

        # --- 2. grounded qualitative research -----------------------------
        res_prompt = (
            _load_prompt("enrich_research")
            .replace("{ticker}", str(ticker or ""))
            .replace("{company_name}", company)
            .replace("{market}", market)
            .replace("{lookback_days}", str(config.research_lookback_days))
        )
        try:
            resp = await _call_with_retry(
                lambda: client.chat.completions.create(
                    model=config.perplexity_model,
                    temperature=0.2,
                    messages=[{"role": "user", "content": res_prompt}],
                ),
                what=f"enrich.research[{ticker}]",
            )
            summary = (resp.choices[0].message.content or "").strip()
            if summary:
                raw = candidate.setdefault("raw_data", {})
                raw["research"] = {
                    "summary": summary,
                    "citations": _extract_citations(resp),
                    "retrieved_at": _utcnow_iso(),
                }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"enrich: {ticker} research failed: {exc}")

    return candidate, errors


def _dedup_for_enrich(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], int]:
    """Drop candidates enriched recently + collapse intra-run duplicates.

    Returns (to_enrich, skipped_count). Mirrors score_blueprint._dedup_for_scoring
    but keys off ``last_enriched`` and ``research_lookback_days``.
    """
    recent_ids = get_recently_enriched_ids(config.db_path, config.research_lookback_days)
    seen: set[str] = set()
    to_enrich: List[Candidate] = []
    skipped = 0
    for cand in candidates:
        cid = cand.get("candidate_id")
        if cid in recent_ids or cid in seen:
            skipped += 1
            continue
        seen.add(cid)
        to_enrich.append(cand)
    return to_enrich, skipped


async def enrich_candidates(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    """Enrich a batch of candidates concurrently. Returns (enriched, errors).

    Graceful degradation: with no PERPLEXITY_API_KEY the stage is a no-op and the
    agent runs exactly as before.
    """
    if not config.perplexity_api_key:
        logger.info("enrich: skipped — PERPLEXITY_API_KEY not set")
        return [], []

    to_enrich, skipped = _dedup_for_enrich(config, candidates)
    capped = to_enrich[: config.max_enrich_per_run]
    logger.info(
        "enrich: %d in, %d skipped (recent/dup), %d to enrich (cap %d)",
        len(candidates), skipped, len(capped), config.max_enrich_per_run,
    )
    if not capped:
        return [], []

    client = AsyncOpenAI(
        api_key=config.perplexity_api_key, base_url=_PERPLEXITY_BASE_URL
    )
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)

    results = await asyncio.gather(
        *(_enrich_one(client, config, semaphore, c) for c in capped)
    )

    enriched: List[Candidate] = []
    errors: List[str] = []
    for candidate, errs in results:
        errors.extend(errs)
        raw = candidate.get("raw_data") or {}
        # Persist (and stamp last_enriched) only when we actually attached data.
        if raw.get("fundamentals") or raw.get("research"):
            try:
                upsert_candidate(config.db_path, candidate)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"enrich: db upsert failed for {candidate.get('ticker')}: {exc}"
                )
            enriched.append(candidate)

    return enriched, errors


def make_enrich_node(config: Config):
    """Factory: returns an async graph node that closes over ``config``."""

    async def _node(state: dict) -> dict:
        enriched, errors = await enrich_candidates(config, state.get("candidates", []))
        logger.info("enrich: attached data to %d candidate(s)", len(enriched))
        # See module docstring: mutate in place, return only errors (the
        # candidates channel must not be re-emitted under its additive reducer).
        return {"errors": errors}

    return _node
