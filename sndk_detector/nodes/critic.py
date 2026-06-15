"""critic node: adversarial pass that tries to break each memo'd thesis.

Runs only on candidates that got a memo. Persists the critic alongside the memo
so the deep-dive output and Telegram alert can surface the single most-likely
invalidating metric — turning the operator's skepticism into a system feature.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Tuple

from ..config import Config
from ..db import upsert_memo
from ..schemas import Critic
from ..state import Candidate
from ._llm import extract_model, get_client, load_prompt, render_candidate_block

logger = logging.getLogger(__name__)


async def _critic_one(
    client, config: Config, semaphore: asyncio.Semaphore, candidate: Candidate
) -> Tuple[Candidate, List[str]]:
    errors: List[str] = []
    cid = candidate["candidate_id"]
    ticker = candidate.get("ticker")
    prompt = load_prompt("critic").replace(
        "{candidate_block}", render_candidate_block(candidate)
    )
    async with semaphore:
        try:
            critic: Critic = await extract_model(
                client, config, prompt, Critic, what=f"critic[{ticker}]", temperature=0.4,
            )
            candidate["critic"] = critic.model_dump()
            # Persist both so we never wipe the memo (upsert overwrites both cols).
            upsert_memo(config.db_path, cid, {
                "memo_json": json.dumps(candidate.get("memo")),
                "critic_json": json.dumps(candidate["critic"]),
                "memo_version": config.scorer_version,
            })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"critic: {ticker} failed: {exc}")
    return candidate, errors


async def critique_candidates(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    has_memo = [c for c in candidates if c.get("memo")]
    if not has_memo:
        return candidates, []

    client = get_client(config)
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)
    results = await asyncio.gather(
        *(_critic_one(client, config, semaphore, c) for c in has_memo)
    )
    errors: List[str] = []
    for _, errs in results:
        errors.extend(errs)
    logger.info("critic: reviewed %d memo'd candidates", len(has_memo))
    return candidates, errors


def make_critic_node(config: Config):
    """Factory: returns an async graph node that closes over ``config``."""

    async def _node(state: dict) -> dict:
        out, errors = await critique_candidates(
            config, state.get("scored_candidates", [])
        )
        return {"scored_candidates": out, "errors": errors}

    return _node
