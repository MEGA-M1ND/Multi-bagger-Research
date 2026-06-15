"""classify_event node: tag the structural event, drop non-spinoffs.

The first LLM pass and the gate for the vertical slice. For each normalized
candidate it classifies the event_family from the filing text and extracts the
key spinoff facts (parent, spun entity, record date, ratio, rationale) as an
EventSignal. Candidates that are not spinoffs/carve-outs are dropped here so we
never spend extraction/scoring tokens on them.

Async fan-out with a semaphore, mirroring score_blueprint.py's concurrency.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional, Tuple

from ..config import Config
from ..db import get_recent_scored_ids, upsert_candidate, upsert_evidence
from ..schemas import EventSignal
from ..state import Candidate
from ._llm import (
    evidence_rows,
    extract_model,
    get_client,
    guard_snippets,
    load_prompt,
    render_candidate_block,
    source_texts_of,
)

logger = logging.getLogger(__name__)

# event_family values we keep for this slice.
_KEPT_FAMILIES = {"spinoff", "carveout"}


async def _classify_one(
    client, config: Config, semaphore: asyncio.Semaphore, candidate: Candidate
) -> Tuple[Candidate, bool, Optional[str]]:
    """Classify one candidate. Returns (candidate, keep, error_or_None)."""
    prompt = load_prompt("event_classifier").replace(
        "{candidate_block}", render_candidate_block(candidate)
    )
    async with semaphore:
        try:
            signal: EventSignal = await extract_model(
                client, config, prompt, EventSignal,
                what=f"classify[{candidate.get('ticker')}]",
            )
        except Exception as exc:  # noqa: BLE001 - record, don't crash the graph
            return candidate, False, f"classify: {candidate.get('ticker')} failed: {exc}"

    # Demote any hallucinated snippets to confidence 0.
    guard_snippets(signal, source_texts_of(candidate))

    candidate["event_family"] = signal.event_family
    candidate["event_signal"] = signal.model_dump()

    # Persist event_signal evidence (provenance llm_filing).
    for row in evidence_rows(candidate["candidate_id"], "event_signal", signal):
        try:
            upsert_evidence(config.db_path, candidate["candidate_id"], row)
        except Exception as exc:  # noqa: BLE001
            return candidate, False, f"classify: {candidate.get('ticker')} evidence persist failed: {exc}"

    keep = signal.event_family in _KEPT_FAMILIES
    return candidate, keep, None


async def classify_candidates(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    """Classify a batch, keeping only spinoffs/carve-outs. Returns (kept, errors)."""
    if not candidates:
        return [], []

    # Token discipline: skip candidates scored within the lookback window BEFORE
    # spending any LLM tokens on classification/extraction.
    recent = get_recent_scored_ids(config.db_path, config.dedup_lookback_days)
    fresh = [c for c in candidates if c.get("candidate_id") not in recent]
    skipped = len(candidates) - len(fresh)
    if skipped:
        logger.info("classify_event: %d skipped (scored within lookback)", skipped)
    if not fresh:
        return [], []
    candidates = fresh

    client = get_client(config)
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)
    results = await asyncio.gather(
        *(_classify_one(client, config, semaphore, c) for c in candidates)
    )

    kept: List[Candidate] = []
    errors: List[str] = []
    dropped = 0
    for candidate, keep, err in results:
        if err:
            errors.append(err)
        # Persist event_family/status on the candidate either way (observability).
        try:
            upsert_candidate(config.db_path, candidate)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"classify: db upsert failed for {candidate.get('ticker')}: {exc}")
        if keep:
            kept.append(candidate)
        else:
            dropped += 1

    logger.info(
        "classify_event: %d in, %d kept (spinoff/carveout), %d dropped",
        len(candidates), len(kept), dropped,
    )
    return kept, errors


def make_classify_event_node(config: Config):
    """Factory: returns an async graph node that closes over ``config``."""

    async def _node(state: dict) -> dict:
        kept, errors = await classify_candidates(
            config, state.get("normalized_candidates", [])
        )
        return {"classified_candidates": kept, "errors": errors}

    return _node
