"""Ingestion: financial news via Finnhub.

Requires a free Finnhub API key (https://finnhub.io — no credit card needed).
Set FINNHUB_API_KEY in .env. Without it, this node returns no candidates.

Strategy:
  1. Fetch general market news + merger/acquisition news from Finnhub.
  2. Filter articles whose headline or summary contains at least one
     blueprint keyword (structural events, secular tailwinds, moat signals).
  3. Turn each matching article with a known ticker into a Candidate.

Finnhub free tier: 60 calls/minute, which is more than enough here.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List, Tuple

import requests

from ..config import Config
from ..state import Candidate, new_candidate

logger = logging.getLogger(__name__)

_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"
_REQUEST_TIMEOUT = 15
_MIN_INTERVAL_SEC = 0.5  # stay well under Finnhub's 60/min free limit

# News categories to pull — "general" covers broad market, "merger" covers M&A
_CATEGORIES = ("general", "merger")

# Max article age in seconds (3 days) — only surface recent news
_MAX_AGE_SECS = 3 * 24 * 3600

# Keywords mapped to blueprint factors. An article matching ANY of these is
# worth turning into a candidate for LLM scoring.
_BLUEPRINT_KEYWORDS = (
    # Factor 1 — Structural event
    "spin-off", "spinoff", "carve-out", "separation", "divestiture",
    "strategic alternative", "restructur", "merger", "acquisition",
    # Factor 3 — Secular tailwind
    "artificial intelligence", "ai chip", "semiconductor", "hyperscaler",
    "small modular reactor", "nuclear", "smr", "defense ai",
    "quantum computing", "robotics", "autonomous",
    # Factor 4 — Supply constraint
    "critical mineral", "sole source", "rare earth", "proprietary",
    # Factor 2 — Cyclical trough
    "cyclical low", "trough", "downcycle", "inventory correction",
)


def _matches_blueprint(headline: str, summary: str) -> bool:
    combined = (headline + " " + summary).lower()
    return any(kw in combined for kw in _BLUEPRINT_KEYWORDS)


def _article_to_candidate(article: dict) -> Candidate | None:
    """Turn a Finnhub news article into a Candidate, or None if unusable."""
    ticker = (article.get("related") or "").strip().upper()
    if not ticker:
        return None
    # Finnhub sometimes puts multiple comma-separated tickers in `related`
    ticker = ticker.split(",")[0].strip()
    if not ticker or len(ticker) > 8:
        return None

    headline = article.get("headline") or ""
    summary = article.get("summary") or ""
    source = article.get("source") or ""
    url = article.get("url") or ""
    ts = article.get("datetime") or 0

    return new_candidate(
        ticker=ticker,
        company_name=ticker,  # Finnhub news doesn't always have a company name
        market="US",
        source="news",
        raw_data={
            "headline": headline,
            "summary": summary[:500],
            "news_source": source,
            "url": url,
            "published_ts": ts,
        },
    )


def fetch_news_candidates(config: Config) -> Tuple[List[Candidate], List[str]]:
    """Return (candidates, errors). Requires FINNHUB_API_KEY in config."""
    if not config.finnhub_api_key:
        note = "ingest_news: FINNHUB_API_KEY not set — skipping news ingest (see .env.example)."
        logger.info(note)
        return [], [note]

    now_ts = datetime.now(timezone.utc).timestamp()
    candidates: List[Candidate] = []
    errors: List[str] = []
    seen_ids: set[str] = set()

    for category in _CATEGORIES:
        if len(candidates) >= config.max_candidates_per_source:
            break
        params = {"category": category, "token": config.finnhub_api_key}
        try:
            time.sleep(_MIN_INTERVAL_SEC)
            resp = requests.get(_FINNHUB_NEWS_URL, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            articles = resp.json()
        except requests.RequestException as exc:
            errors.append(f"ingest_news: request failed for category={category}: {exc}")
            continue
        except ValueError as exc:
            errors.append(f"ingest_news: invalid JSON for category={category}: {exc}")
            continue

        if not isinstance(articles, list):
            errors.append(f"ingest_news: unexpected response shape for category={category}")
            continue

        for article in articles:
            if len(candidates) >= config.max_candidates_per_source:
                break
            # Skip stale articles
            age = now_ts - (article.get("datetime") or 0)
            if age > _MAX_AGE_SECS:
                continue

            headline = article.get("headline") or ""
            summary = article.get("summary") or ""
            if not _matches_blueprint(headline, summary):
                continue

            try:
                cand = _article_to_candidate(article)
            except Exception as exc:
                errors.append(f"ingest_news: failed to parse article: {exc}")
                continue

            if cand is None:
                continue
            if cand["candidate_id"] in seen_ids:
                continue
            seen_ids.add(cand["candidate_id"])
            candidates.append(cand)

    logger.info("ingest_news: %d candidates, %d errors", len(candidates), len(errors))
    return candidates, errors


def make_ingest_news_node(config: Config):
    """Factory: returns a graph node that closes over config."""

    def _node(state: dict) -> dict:
        candidates, errors = fetch_news_candidates(config)
        return {"candidates": candidates, "errors": errors}

    return _node
