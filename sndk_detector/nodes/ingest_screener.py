"""Ingestion: India equities screener (STUB).

An honest stub for India-listed candidates (NSE/BSE), mirroring the other
ingestion sources. Returns no candidates and a note in ``errors``; it does NOT
fabricate data.

To implement: query a screener/data source (e.g. screener.in, NSE bhavcopy,
Tickertape, or a broker API), filter for the kind of names the blueprint cares
about, and emit Candidates with ``market="IN"`` and ``source="screener_in"``.
Keep the try/except discipline and honor ``config.max_candidates_per_source``.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..config import Config
from ..state import Candidate

logger = logging.getLogger(__name__)


def fetch_screener_candidates(config: Config) -> Tuple[List[Candidate], List[str]]:
    """Return (candidates, errors). STUB — returns no candidates.

    TODO: implement against a real India screener/data source. Suggested shape:

        candidates = []
        for row in run_screen(...):               # network call in try/except
            candidates.append(new_candidate(
                ticker=row.symbol,
                company_name=row.name,
                market="IN",
                source="screener_in",
                price=row.price,
                market_cap=row.market_cap,
                raw_data={"pe": row.pe, "sector": row.sector, ...},
            ))
            if len(candidates) >= config.max_candidates_per_source:
                break
        return candidates, errors
    """
    note = (
        "ingest_screener: STUB — no India screener configured yet "
        "(see ingest_screener.py TODO)."
    )
    logger.info(note)
    return [], [note]


def make_ingest_screener_node(config: Config):
    """Factory: returns a graph node that closes over ``config``."""

    def _node(state: dict) -> dict:
        candidates, errors = fetch_screener_candidates(config)
        return {"candidates": candidates, "errors": errors}

    return _node
