"""decide node: route each scored candidate into a decision tier.

No LLM, no new judgment — the tier already lives on the scorecard (set
deterministically by scoring.score, with hard-fail forcing 'reject'). This node
just surfaces it as the candidate's ``status`` and produces the decided list the
output/alert nodes consume.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Tuple

from ..config import Config
from ..state import Candidate

logger = logging.getLogger(__name__)


def decide_candidates(candidates: List[Candidate]) -> Tuple[List[Candidate], Counter]:
    counts: Counter = Counter()
    for cand in candidates:
        tier = (cand.get("scorecard") or {}).get("tier", "reject")
        cand["status"] = tier
        counts[tier] += 1
    return candidates, counts


def make_decide_node(config: Config):
    """Factory: returns a graph node that closes over ``config``."""

    def _node(state: dict) -> dict:
        decided, counts = decide_candidates(state.get("scored_candidates", []))
        logger.info(
            "decide: reject=%d watchlist=%d deep_dive=%d starter=%d",
            counts.get("reject", 0), counts.get("watchlist", 0),
            counts.get("deep_dive", 0), counts.get("starter", 0),
        )
        return {"decided_candidates": decided}

    return _node
