"""Ingestion: financial news (STUB).

This is an honest stub with the same interface as the other ingestion sources.
It returns an empty candidate list and a one-line note in ``errors`` so the run
summary makes it obvious the source isn't wired up yet. It does NOT fabricate
candidates.

To implement: pick a source (e.g. an RSS feed, a news API like NewsAPI/Finnhub,
or scraping a curated list of press-release wires), parse each item into a
Candidate via ``new_candidate(...)``, and respect
``config.max_candidates_per_source``. Keep the try/except discipline: append
failures to the returned errors list, never raise.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..config import Config
from ..state import Candidate

logger = logging.getLogger(__name__)


def fetch_news_candidates(config: Config) -> Tuple[List[Candidate], List[str]]:
    """Return (candidates, errors). STUB — returns no candidates.

    TODO: implement against a real news source. Suggested shape:

        candidates = []
        for item in fetch_feed(...):              # network call in try/except
            candidates.append(new_candidate(
                ticker=item.ticker,
                company_name=item.company,
                market=item.market,               # "US" / "IN" / ...
                source="news",
                raw_data={"headline": item.headline, "url": item.url, ...},
            ))
            if len(candidates) >= config.max_candidates_per_source:
                break
        return candidates, errors
    """
    note = "ingest_news: STUB — no news source configured yet (see ingest_news.py TODO)."
    logger.info(note)
    return [], [note]


def make_ingest_news_node(config: Config):
    """Factory: returns a graph node that closes over ``config``."""

    def _node(state: dict) -> dict:
        candidates, errors = fetch_news_candidates(config)
        return {"candidates": candidates, "errors": errors}

    return _node
