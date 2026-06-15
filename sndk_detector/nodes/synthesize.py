"""synthesize node: write the investment memo — only for surviving candidates.

Gated on tier: we only spend memo tokens on candidates that cleared at least the
watchlist threshold and did NOT hard-fail. The memo is written strictly over the
structured evidence (the prompt forbids new facts); rejected names pass through
untouched.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Tuple

from ..config import Config
from ..db import upsert_memo
from ..schemas import Memo
from ..state import Candidate
from ._llm import extract_model, get_client, load_prompt, render_candidate_block

logger = logging.getLogger(__name__)


def _qualifies(candidate: Candidate) -> bool:
    """Memo-worthy iff scored at watchlist or above (i.e. not reject)."""
    tier = (candidate.get("scorecard") or {}).get("tier")
    return tier in ("watchlist", "deep_dive", "starter")


async def _memo_one(
    client, config: Config, semaphore: asyncio.Semaphore, candidate: Candidate
) -> Tuple[Candidate, List[str]]:
    errors: List[str] = []
    cid = candidate["candidate_id"]
    ticker = candidate.get("ticker")
    prompt = load_prompt("memo_writer").replace(
        "{candidate_block}", render_candidate_block(candidate)
    )
    async with semaphore:
        try:
            memo: Memo = await extract_model(
                client, config, prompt, Memo, what=f"memo[{ticker}]", temperature=0.4,
            )
            candidate["memo"] = memo.model_dump()
            upsert_memo(config.db_path, cid, {
                "memo_json": json.dumps(candidate["memo"]),
                "memo_version": config.scorer_version,
            })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"synthesize: {ticker} memo failed: {exc}")
    return candidate, errors


async def synthesize_candidates(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    qualifying = [c for c in candidates if _qualifies(c)]
    logger.info(
        "synthesize: %d/%d candidates qualify for a memo (>= watchlist)",
        len(qualifying), len(candidates),
    )
    if not qualifying:
        return candidates, []

    client = get_client(config)
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)
    results = await asyncio.gather(
        *(_memo_one(client, config, semaphore, c) for c in qualifying)
    )
    errors: List[str] = []
    for _, errs in results:
        errors.extend(errs)
    # Candidates are enriched in place; return the full list (memo'd + not).
    return candidates, errors


def make_synthesize_node(config: Config):
    """Factory: returns an async graph node that closes over ``config``."""

    async def _node(state: dict) -> dict:
        out, errors = await synthesize_candidates(
            config, state.get("scored_candidates", [])
        )
        return {"scored_candidates": out, "errors": errors}

    return _node
