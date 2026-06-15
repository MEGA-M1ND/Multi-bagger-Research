"""LangGraph state schema for the SNDK Detector pipeline.

The key subtlety here is the use of ``Annotated[List[...], operator.add]`` on
fields that are written by *parallel* nodes. LangGraph treats the annotation as
a reducer: when several ingestion branches run in the same superstep and each
returns ``{"candidates": [...]}``, LangGraph concatenates them instead of having
the last writer win. Without the reducer, parallel fan-out would silently drop
all but one branch's results.
"""

from __future__ import annotations

import hashlib
import operator
from typing import Annotated, Any, List, Optional, TypedDict


class BlueprintScore(TypedDict):
    """The 6-point blueprint, as scored by the LLM.

    Each factor is a boolean; ``total_score`` is the count of True factors
    (0-6). We recompute ``total_score`` from the booleans in code rather than
    trusting the model's arithmetic.
    """

    structural_event: bool
    cyclical_trough: bool
    secular_tailwind: bool
    supply_constraint: bool
    undervalued_narrative: bool
    domain_edge: bool
    total_score: int
    reasoning: str


class Candidate(TypedDict, total=False):
    """A single stock candidate flowing through the pipeline.

    ``total=False`` so ingestion nodes can emit partially-populated candidates
    (no blueprint/thesis yet) and later stages enrich them.
    """

    ticker: str
    company_name: str
    market: str  # e.g. "US", "IN"
    source: str  # e.g. "sec_edgar", "news", "screener_in"
    raw_data: dict  # source-specific payload, kept for auditing/observability
    candidate_id: str  # stable hash of ticker+source, used for dedup/idempotency

    # Enriched during scoring:
    blueprint: Optional[BlueprintScore]
    thesis: Optional[str]
    price: Optional[float]
    market_cap: Optional[float]
    alerted: bool


class AgentState(TypedDict):
    """Top-level graph state.

    ``candidates`` and ``errors`` accumulate across parallel branches (reducer
    = list concatenation). The other fields are written by single, non-parallel
    nodes so they use last-writer-wins (the default).
    """

    candidates: Annotated[List[Candidate], operator.add]
    scored_candidates: List[Candidate]
    alert_queue: List[Candidate]
    run_timestamp: str
    errors: Annotated[List[str], operator.add]


def make_candidate_id(ticker: str, source: str) -> str:
    """Stable, deterministic id for dedup/idempotency.

    Same ticker from the same source always maps to the same id, so re-running
    the agent never creates duplicate rows and the DB can answer "have we seen
    / scored / alerted this before?".
    """
    raw = f"{(ticker or '').strip().upper()}::{(source or '').strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def new_candidate(
    ticker: str,
    company_name: str,
    market: str,
    source: str,
    raw_data: Optional[dict] = None,
    price: Optional[float] = None,
    market_cap: Optional[float] = None,
) -> Candidate:
    """Factory that guarantees a well-formed Candidate with a candidate_id."""
    return Candidate(
        ticker=ticker,
        company_name=company_name,
        market=market,
        source=source,
        raw_data=raw_data or {},
        candidate_id=make_candidate_id(ticker, source),
        blueprint=None,
        thesis=None,
        price=price,
        market_cap=market_cap,
        alerted=False,
    )
