"""Scoring node: score candidates against the 6-point blueprint via the LLM.

Two reasons this is async rather than a plain loop:

  1. Throughput — a run may carry 40+ candidates. We score them concurrently.
  2. Cost control — we bound concurrency with a semaphore (default 5) so we
     don't hammer the API, and we retry with exponential backoff on transient
     rate-limit / server errors.

Token discipline:
  * We dedup against the DB first (skip anything scored within the lookback
    window) so re-runs don't re-pay for the same candidates.
  * A thesis (a second, more expensive call) is only generated for candidates
    that clear BLUEPRINT_THRESHOLD.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from openai import AsyncOpenAI
from openai import APIConnectionError, APIError, RateLimitError

from ..config import Config
from ..db import get_recent_candidate_ids, upsert_candidate
from ..state import BlueprintScore, Candidate

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# The six blueprint factor keys, in order.
_FACTORS = (
    "structural_event",
    "cyclical_trough",
    "secular_tailwind",
    "supply_constraint",
    "undervalued_narrative",
    "domain_edge",
)

_MAX_RETRIES = 4
_BASE_BACKOFF = 1.5  # seconds; grows ~ _BASE_BACKOFF ** attempt


@lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    """Load and cache a prompt template from prompts/."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def _candidate_block(candidate: Candidate) -> str:
    """Render a candidate into a compact, LLM-friendly text block."""
    fields = {
        "ticker": candidate.get("ticker"),
        "company_name": candidate.get("company_name"),
        "market": candidate.get("market"),
        "source": candidate.get("source"),
        "price": candidate.get("price"),
        "market_cap": candidate.get("market_cap"),
        "raw_data": candidate.get("raw_data") or {},
    }
    return json.dumps(fields, indent=2, default=str)


def _coerce_blueprint(parsed: dict) -> BlueprintScore:
    """Turn the model's JSON into a validated BlueprintScore.

    We recompute total_score from the booleans rather than trusting the model's
    arithmetic, and coerce each factor to a real bool.
    """
    booleans = {factor: bool(parsed.get(factor, False)) for factor in _FACTORS}
    total = sum(1 for v in booleans.values() if v)
    return BlueprintScore(
        **booleans,
        total_score=total,
        reasoning=str(parsed.get("reasoning", "")).strip(),
    )


async def _call_with_retry(coro_factory, what: str) -> Optional[object]:
    """Await an OpenAI call, retrying transient failures with backoff.

    ``coro_factory`` is a zero-arg callable returning a fresh awaitable each
    attempt (you can't re-await a spent coroutine). Returns the response, or
    raises the last exception after exhausting retries.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await coro_factory()
        except (RateLimitError, APIConnectionError) as exc:
            last_exc = exc
            # Exponential backoff with jitter to avoid thundering-herd retries.
            delay = _BASE_BACKOFF ** (attempt + 1) + random.uniform(0, 0.5)
            logger.warning(
                "%s: transient error (attempt %d/%d), backing off %.1fs: %s",
                what, attempt + 1, _MAX_RETRIES, delay, exc,
            )
            await asyncio.sleep(delay)
        except APIError as exc:
            # 5xx server errors are worth one or two retries too.
            last_exc = exc
            if attempt >= _MAX_RETRIES - 1:
                break
            delay = _BASE_BACKOFF ** (attempt + 1) + random.uniform(0, 0.5)
            logger.warning("%s: API error, retrying in %.1fs: %s", what, delay, exc)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _score_one(
    client: AsyncOpenAI,
    config: Config,
    semaphore: asyncio.Semaphore,
    candidate: Candidate,
) -> Tuple[Candidate, Optional[str]]:
    """Score a single candidate. Returns (candidate, error_or_None).

    On success the candidate is enriched in-place with ``blueprint`` (and
    ``thesis`` if it clears the threshold). On failure the candidate is returned
    unscored and the error string is non-None.
    """
    scorer_prompt = _load_prompt("blueprint_scorer").replace(
        "{candidate_block}", _candidate_block(candidate)
    )

    # The semaphore is the concurrency cap: at most config.max_concurrent_llm
    # candidates hold it at once, so no more than that many calls are in flight.
    async with semaphore:
        try:
            resp = await _call_with_retry(
                lambda: client.chat.completions.create(
                    model=config.openai_model,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    messages=[{"role": "user", "content": scorer_prompt}],
                ),
                what=f"score[{candidate.get('ticker')}]",
            )
        except Exception as exc:  # noqa: BLE001 - report, don't crash the graph
            return candidate, f"score: {candidate.get('ticker')} failed: {exc}"

        try:
            content = resp.choices[0].message.content or "{}"
            blueprint = _coerce_blueprint(json.loads(content))
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            return candidate, f"score: {candidate.get('ticker')} bad JSON: {exc}"

        candidate["blueprint"] = blueprint

        # Only spend a second call on candidates that clear the bar.
        if blueprint["total_score"] >= config.blueprint_threshold:
            thesis_err = await _generate_thesis(client, config, semaphore, candidate)
            if thesis_err:
                return candidate, thesis_err  # scored, but thesis failed

    return candidate, None


async def _generate_thesis(
    client: AsyncOpenAI,
    config: Config,
    semaphore: asyncio.Semaphore,
    candidate: Candidate,
) -> Optional[str]:
    """Generate and attach a thesis. Returns an error string or None.

    NOTE: called from inside ``_score_one`` which already holds the semaphore,
    so we do NOT re-acquire it here (that would deadlock at max concurrency).
    """
    blueprint = candidate.get("blueprint") or {}
    prompt = (
        _load_prompt("thesis_generator")
        .replace("{candidate_block}", _candidate_block(candidate))
        .replace("{blueprint_block}", json.dumps(blueprint, indent=2, default=str))
    )
    try:
        resp = await _call_with_retry(
            lambda: client.chat.completions.create(
                model=config.openai_model,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            ),
            what=f"thesis[{candidate.get('ticker')}]",
        )
        candidate["thesis"] = (resp.choices[0].message.content or "").strip()
        return None
    except Exception as exc:  # noqa: BLE001
        return f"thesis: {candidate.get('ticker')} failed: {exc}"


def _dedup_for_scoring(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], int]:
    """Drop candidates already scored recently + collapse intra-run duplicates.

    Returns (to_score, skipped_count).
    """
    recent_ids = get_recent_candidate_ids(config.db_path, config.dedup_lookback_days)
    seen: set[str] = set()
    to_score: List[Candidate] = []
    skipped = 0
    for cand in candidates:
        cid = cand["candidate_id"]
        if cid in recent_ids or cid in seen:
            skipped += 1
            continue
        seen.add(cid)
        to_score.append(cand)
    return to_score, skipped


async def score_candidates(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    """Score a batch of candidates concurrently. Returns (scored, errors)."""
    to_score, skipped = _dedup_for_scoring(config, candidates)
    logger.info(
        "score: %d candidates in, %d skipped (recent/dup), %d to score",
        len(candidates), skipped, len(to_score),
    )
    if not to_score:
        return [], []

    client = AsyncOpenAI(api_key=config.openai_api_key)
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)

    # Fan out all scoring calls at once; the semaphore throttles actual
    # concurrency. return_exceptions=False because _score_one already catches
    # everything and returns errors as data.
    results = await asyncio.gather(
        *(_score_one(client, config, semaphore, c) for c in to_score)
    )

    scored: List[Candidate] = []
    errors: List[str] = []
    for candidate, err in results:
        if err:
            errors.append(err)
        # Persist anything that actually got a blueprint (even if thesis failed).
        if candidate.get("blueprint"):
            try:
                upsert_candidate(config.db_path, candidate)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"score: db upsert failed for {candidate.get('ticker')}: {exc}")
            scored.append(candidate)

    return scored, errors


def make_score_node(config: Config):
    """Factory: returns an async graph node that closes over ``config``."""

    async def _node(state: dict) -> dict:
        scored, errors = await score_candidates(config, state.get("candidates", []))
        return {"scored_candidates": scored, "errors": errors}

    return _node
