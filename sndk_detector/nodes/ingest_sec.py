"""Ingestion: SEC EDGAR full-text search.

Endpoint (the JSON backend behind https://efts.sec.gov/LATEST/search-index):

    https://efts.sec.gov/LATEST/search-index?q="spin-off"&forms=8-K,S-1,10-12B

We search for structural-event language (spin-offs, carve-outs, etc.) across the
form types most likely to announce them, and turn each hit into a Candidate.

------------------------------------------------------------------------------
RESPONSE SHAPE — IMPORTANT
------------------------------------------------------------------------------
EDGAR full-text search is an Elasticsearch backend. The response shape used by
the parser below is the *documented / stable* one:

    {
      "hits": {
        "total": {"value": <int>},
        "hits": [
          {
            "_id": "<accession>:<filename>",
            "_source": {
              "display_names": ["Company Name (TICKER) (CIK 000...)"],
              "ciks": ["0000320193"],
              "form": "8-K",
              "root_forms": ["8-K"],
              "file_date": "2024-01-01",
              "file_type": "8-K",
              "file_description": "...",
              "biz_states": ["CA"],
              "sics": ["3571"]
            }
          },
          ...
        ]
      }
    }

This parser was written WITHOUT a live verification call: the SEC hosts
(efts.sec.gov / data.sec.gov) are blocked by this environment's network egress
allowlist, so the one real call the build spec asked for returned HTTP 403
("Host not in allowlist"). To stay honest about that:

  * ``_extract_candidate`` is fully defensive — every field access uses ``.get``
    and tolerates missing/renamed keys, returning None rather than raising.
  * ``_warn_if_unexpected_shape`` logs (once) if the top-level shape doesn't
    look like what we expect, so a drift in the live API surfaces loudly in the
    run errors instead of silently yielding zero candidates.

If you can reach SEC and the shape differs, paste a sample response and adjust
``_extract_candidate`` — that's the only function that needs to change.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import List, Optional, Tuple

import requests

from ..config import Config
from ..state import Candidate, new_candidate

logger = logging.getLogger(__name__)

SEC_FTS_URL = "https://efts.sec.gov/LATEST/search-index"

# Only look at filings from the last N days — keeps candidates current.
_LOOKBACK_DAYS = 60
# Max hits to accept per query so every query group gets a fair share of the
# overall MAX_CANDIDATES_PER_SOURCE cap (instead of the first query eating it).
_MAX_PER_QUERY = 5

# v2 SPINOFF VERTICAL SLICE: this ingestion is now scoped to ONE event family —
# spinoffs / carve-outs. Every query is spinoff-distribution language; the forms
# are the ones that announce or register a separation. Other event families
# (activist, recap, etc.) come in a later phase.
SEC_QUERIES = (
    '"spin-off"',
    '"spinoff"',
    '"carve-out"',
    '"separation of"',
    '"information statement"',
    '"pro rata distribution"',
    '"distribution of"',
)
# 10-12B = registration of a spun entity's securities (the canonical spinoff
# form); 8-K announces the separation; S-1 covers carve-out IPOs.
SEC_FORMS = "10-12B,8-K,S-1"

# SEC asks for <= 10 requests/sec. We make very few requests, but be polite.
_MIN_INTERVAL_SEC = 0.12
_REQUEST_TIMEOUT = 20

# Pull "TICKER" out of an EDGAR display name like "Acme Corp (ACME) (CIK 000...)".
_TICKER_RE = re.compile(r"\(([A-Z0-9.\-]{1,8})\)")

# Max chars to keep from a fetched filing — enough context without flooding the LLM.
_FILING_SNIPPET_CHARS = 2500


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text stripper using stdlib only."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(raw: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(raw)
    except Exception:
        pass
    return parser.get_text()


def _source_id(accession: str, filename: str) -> str:
    """Stable id for a source document (matches db.upsert_source_document)."""
    return hashlib.sha256(f"{accession}:{filename}".encode("utf-8")).hexdigest()[:16]


def _build_source_document(hit: dict, headers: dict) -> Optional[dict]:
    """Fetch a filing's opening text + provenance metadata.

    Returns a source_document dict (source_id, cik, form, accession, filename,
    url, file_date, fetched_text) or None on any failure. Non-fatal by design —
    a missing doc doesn't break the pipeline, it just leaves less evidence.
    """
    hit_id = hit.get("_id") or ""
    parts = hit_id.split(":", 1)
    if len(parts) < 2 or not parts[1]:
        return None
    accession, filename = parts[0], parts[1]
    # Skip index pages — they contain links, not filing content
    if "index" in filename.lower():
        return None

    source_doc = (hit.get("_source") or {})
    ciks = source_doc.get("ciks") or []
    if not ciks:
        return None

    cik_int = str(int(ciks[0]))  # strip leading zeros
    accession_nodashes = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_nodashes}/{filename}"
    )
    fetched_text = None
    try:
        time.sleep(_MIN_INTERVAL_SEC)
        resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type or filename.endswith((".htm", ".html")):
            text = _strip_html(resp.text)
        else:
            text = resp.text
        fetched_text = text[:_FILING_SNIPPET_CHARS].strip() or None
    except Exception:
        fetched_text = None

    return {
        "source_id": _source_id(accession, filename),
        "cik": str(ciks[0]),
        "form": source_doc.get("form") or source_doc.get("file_type"),
        "accession": accession,
        "filename": filename,
        "url": url,
        "file_date": source_doc.get("file_date"),
        "fetched_text": fetched_text,
    }


def _polite_get(url: str, params: dict, headers: dict) -> requests.Response:
    """A single rate-limited GET. Caller handles exceptions."""
    time.sleep(_MIN_INTERVAL_SEC)  # crude but sufficient rate limiting
    resp = requests.get(url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def _warn_if_unexpected_shape(payload: dict, errors: List[str]) -> None:
    """Record a soft warning if the response isn't shaped like we expect."""
    if not isinstance(payload, dict) or "hits" not in payload:
        errors.append(
            "ingest_sec: unexpected SEC response shape (no top-level 'hits'); "
            "parser may need updating — see ingest_sec.py docstring."
        )


def _parse_display_name(display_name: str) -> Tuple[Optional[str], str]:
    """Split an EDGAR display name into (ticker, company_name).

    Display names look like "Acme Corp (ACME) (CIK 0001234567)". Many filers
    have no ticker, in which case ticker is None and we fall back to a CIK-ish
    placeholder so the candidate still has a stable identity.
    """
    company = display_name.strip()
    ticker = None
    match = _TICKER_RE.search(display_name)
    if match:
        candidate_ticker = match.group(1)
        # Skip the "(CIK ...)" group — it won't match the short-ticker pattern,
        # but guard anyway.
        if candidate_ticker.upper() != "CIK":
            ticker = candidate_ticker
    # Trim everything from the first " (" so company_name is just the name.
    paren = company.find(" (")
    if paren != -1:
        company = company[:paren].strip()
    return ticker, company


def _extract_candidate(hit: dict, query: str) -> Optional[Candidate]:
    """Turn one ES hit into a Candidate, or None if it's unusable.

    Defensive throughout: any missing field degrades gracefully.
    """
    source_doc = (hit or {}).get("_source") or {}
    display_names = source_doc.get("display_names") or []
    if not display_names:
        return None

    ticker, company = _parse_display_name(str(display_names[0]))

    # Fall back to CIK for identity when there's no ticker (common for filers).
    ciks = source_doc.get("ciks") or []
    if not ticker:
        ticker = f"CIK{ciks[0]}" if ciks else None
    if not ticker:
        return None

    raw = {
        "accession": (hit.get("_id") or "").split(":")[0],
        "form": source_doc.get("form") or source_doc.get("file_type"),
        "root_forms": source_doc.get("root_forms"),
        "file_date": source_doc.get("file_date"),
        "file_description": source_doc.get("file_description"),
        "ciks": ciks,
        "biz_states": source_doc.get("biz_states"),
        "matched_query": query,
        "display_name": display_names[0],
    }

    cand = new_candidate(
        ticker=ticker,
        company_name=company or ticker,
        market="US",
        source="sec_edgar",
        raw_data=raw,
    )
    # v2: carry the CIK as first-class identity (normalize/extract rely on it).
    if ciks:
        cand["cik"] = str(ciks[0])
    return cand


def fetch_sec_candidates(config: Config) -> Tuple[List[Candidate], List[str]]:
    """Fetch SEC EDGAR candidates. Returns (candidates, errors).

    Never raises: all network/parse failures are captured into ``errors`` so the
    graph keeps running even if SEC is unreachable.
    """
    headers = {
        # SEC REQUIRES a descriptive User-Agent with contact info or returns 403.
        "User-Agent": config.sec_user_agent,
        "Accept": "application/json",
    }

    candidates: List[Candidate] = []
    errors: List[str] = []
    seen_ids: set[str] = set()

    now = datetime.now(timezone.utc)
    start_dt = (now - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_dt = now.strftime("%Y-%m-%d")

    for query in SEC_QUERIES:
        if len(candidates) >= config.max_candidates_per_source:
            break
        params = {
            "q": query,
            "forms": SEC_FORMS,
            "dateRange": "custom",
            "startdt": start_dt,
            "enddt": end_dt,
        }
        try:
            resp = _polite_get(SEC_FTS_URL, params, headers)
            payload = resp.json()
        except requests.RequestException as exc:
            errors.append(f"ingest_sec: request failed for q={query}: {exc}")
            continue
        except ValueError as exc:  # JSON decode error
            errors.append(f"ingest_sec: invalid JSON for q={query}: {exc}")
            continue

        _warn_if_unexpected_shape(payload, errors)
        hits = (((payload or {}).get("hits") or {}).get("hits")) or []

        query_count = 0
        for hit in hits:
            if len(candidates) >= config.max_candidates_per_source:
                break
            if query_count >= _MAX_PER_QUERY:
                break
            try:
                cand = _extract_candidate(hit, query)
            except Exception as exc:  # never let one bad hit kill the batch
                errors.append(f"ingest_sec: failed to parse a hit: {exc}")
                continue
            if cand is None:
                continue
            # Dedup within this run (same filer matching multiple queries).
            if cand["candidate_id"] in seen_ids:
                continue
            seen_ids.add(cand["candidate_id"])
            # Build the source_document (provenance + filing text). normalize
            # persists it to the DB; extract_evidence reads its fetched_text.
            doc = _build_source_document(hit, headers)
            if doc:
                raw = cand.get("raw_data") or {}
                raw["source_documents"] = [doc]
                # Keep filing_text top-level for back-compat with current scorer.
                if doc.get("fetched_text"):
                    raw["filing_text"] = doc["fetched_text"]
                cand["raw_data"] = raw
            candidates.append(cand)
            query_count += 1

    return candidates, errors


def make_ingest_sec_node(config: Config):
    """Factory: returns a graph node that closes over ``config``."""

    def _node(state: dict) -> dict:
        candidates, errors = fetch_sec_candidates(config)
        logger.info("ingest_sec: %d candidates, %d errors", len(candidates), len(errors))
        return {"candidates": candidates, "errors": errors}

    return _node
