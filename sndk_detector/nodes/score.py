"""score node: deterministic valuation + 100-point weighted scorecard.

No LLM here — by design. This node computes the peer-relative valuation gap
(yfinance, run in a thread), then scores the candidate mechanically from the
evidence the earlier nodes attached. A hard fail (from the hard_fail node) forces
tier='reject' regardless of the points.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

from ..config import Config
from ..db import upsert_candidate, upsert_evidence, upsert_scorecard
from ..scoring import compute_priority, score
from ..state import Candidate
from ..valuation import compute_valuation_gap
from ._llm import evidence_rows
from ..schemas import ValuationGap

logger = logging.getLogger(__name__)


async def _score_one(
    config: Config, semaphore: asyncio.Semaphore, candidate: Candidate
) -> Tuple[Candidate, List[str]]:
    errors: List[str] = []
    cid = candidate["candidate_id"]
    ticker = candidate.get("ticker")

    # --- valuation gap (deterministic; peer fetch hits the network) ---
    async with semaphore:
        try:
            gap: ValuationGap = await asyncio.to_thread(compute_valuation_gap, candidate)
            candidate["valuation_gap"] = gap.model_dump()
            for row in evidence_rows(cid, "valuation_gap", gap, provenance="deterministic"):
                upsert_evidence(config.db_path, cid, row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"score: {ticker} valuation failed: {exc}")

    # --- priority overlay + weighted score (pure) ---
    candidate["priority_for_me"] = compute_priority(candidate)
    thresholds = (config.tier_watchlist, config.tier_deep_dive, config.tier_starter)
    card = score(candidate, thresholds, config.scorer_version)
    scorecard = {**card.model_dump(), "scorer_version": config.scorer_version}
    candidate["scorecard"] = scorecard
    candidate["status"] = card.tier

    try:
        upsert_scorecard(config.db_path, cid, scorecard)
        upsert_candidate(config.db_path, candidate)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"score: {ticker} persist failed: {exc}")

    return candidate, errors


async def score_candidates(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    if not candidates:
        return [], []
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)
    results = await asyncio.gather(
        *(_score_one(config, semaphore, c) for c in candidates)
    )
    scored: List[Candidate] = []
    errors: List[str] = []
    for candidate, errs in results:
        errors.extend(errs)
        scored.append(candidate)
    logger.info("score: scored %d candidates", len(scored))
    return scored, errors


def make_score_node(config: Config):
    """Factory: returns an async graph node that closes over ``config``."""

    async def _node(state: dict) -> dict:
        scored, errors = await score_candidates(
            config, state.get("enriched_candidates", [])
        )
        return {"scored_candidates": scored, "errors": errors}

    return _node
