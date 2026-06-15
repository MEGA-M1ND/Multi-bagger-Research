"""hard_fail node: deterministic + LLM disqualifier gate.

Runs BEFORE scoring and synthesis so we drop the names that should die early —
the spinoffs that look exciting but can't survive (refinancing wall, dilution,
governance rot, no catalyst). A hard fail does NOT remove the candidate from the
pipeline (we still score it for observability), but it forces tier='reject' and
suppresses the expensive memo/critic synthesis downstream.

The deterministic checks (cash runway, leverage) live in scoring.evaluate_hard_fails;
this node adds the LLM-extracted RiskFlag evidence for risks that aren't visible
in the numbers alone, then combines them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

from ..config import Config
from ..db import upsert_evidence
from ..schemas import RiskFlag
from ..scoring import evaluate_hard_fails
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


async def _check_one(
    client, config: Config, semaphore: asyncio.Semaphore, candidate: Candidate
) -> Tuple[Candidate, List[str]]:
    errors: List[str] = []
    cid = candidate["candidate_id"]
    ticker = candidate.get("ticker")

    risk = RiskFlag()
    prompt = load_prompt("risk_disqualifier").replace(
        "{candidate_block}", render_candidate_block(candidate)
    )
    async with semaphore:
        try:
            risk = await extract_model(
                client, config, prompt, RiskFlag, what=f"risk[{ticker}]",
            )
            guard_snippets(risk, source_texts_of(candidate))
            for row in evidence_rows(cid, "risk_flag", risk, provenance="llm_filing"):
                upsert_evidence(config.db_path, cid, row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"hard_fail: {ticker} risk extraction failed: {exc}")

    risk_dump = risk.model_dump()
    hard_fail, reasons = evaluate_hard_fails(
        candidate.get("financial_snapshot"),
        candidate.get("moat_proxy"),
        risk_dump,
    )
    candidate["risk_flags"] = {
        "flags": risk_dump,
        "hard_fail": hard_fail,
        "reasons": reasons,
    }
    if hard_fail:
        logger.info("hard_fail: %s -> REJECT (%s)", ticker, "; ".join(reasons))
    return candidate, errors


async def run_hard_fail(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    if not candidates:
        return [], []
    client = get_client(config)
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)
    results = await asyncio.gather(
        *(_check_one(client, config, semaphore, c) for c in candidates)
    )
    out: List[Candidate] = []
    errors: List[str] = []
    failed = 0
    for candidate, errs in results:
        errors.extend(errs)
        if candidate.get("risk_flags", {}).get("hard_fail"):
            failed += 1
        out.append(candidate)
    logger.info("hard_fail: %d checked, %d hard-failed", len(out), failed)
    return out, errors


def make_hard_fail_node(config: Config):
    """Factory: returns an async graph node that closes over ``config``."""

    async def _node(state: dict) -> dict:
        # hard_fail enriches candidates in place; they continue to scoring under
        # the same 'enriched_candidates' key (last-writer-wins, no reducer).
        out, errors = await run_hard_fail(config, state.get("enriched_candidates", []))
        return {"enriched_candidates": out, "errors": errors}

    return _node
