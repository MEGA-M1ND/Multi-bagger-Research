"""extract_evidence node: hybrid fundamentals + moat extraction.

For each surviving spinoff candidate:
  1. Pull fundamentals from yfinance (fast, free, high base confidence).
  2. If a required field is missing/sparse (common for freshly-spun entities
     with no trading history), fall back to LLM extraction from the filing text,
     recording the exact source snippet + a rubric confidence.
  3. Extract hard moat proxies from the filing text via the LLM.

All evidence is persisted with explicit provenance ('yfinance' | 'llm_filing')
so a downstream memo can show whether a number came from market data or a quoted
filing sentence. Defensive throughout; failures become error strings.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from typing import Dict, List, Optional, Tuple

from ..config import Config
from ..db import upsert_candidate, upsert_evidence
from ..schemas import EvidenceField, FinancialSnapshot, MoatProxy
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

# yfinance gets a solid base confidence; it's structured market data, not prose.
_YF_CONFIDENCE = 0.85
# Fields that must be present or we trigger the LLM filing fallback.
_REQUIRED = ("revenue_ttm", "net_debt", "cash", "shares_out")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


def _num(x) -> Optional[float]:
    """Coerce to float, treating None/NaN/non-numeric as missing."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _yf_financials(ticker: str) -> Tuple[FinancialSnapshot, Dict, set]:
    """Build a FinancialSnapshot from yfinance .info. Synchronous (run in a thread).

    Returns (snapshot, yf_data_for_valuation, set_of_filled_field_names).
    """
    t = yf.Ticker(ticker)
    info = t.info or {}

    revenue = _num(info.get("totalRevenue"))
    ebitda = _num(info.get("ebitda"))
    gross_margin = _num(info.get("grossMargins"))
    cash = _num(info.get("totalCash"))
    total_debt = _num(info.get("totalDebt"))
    net_debt = (total_debt - cash) if (total_debt is not None and cash is not None) else None
    fcf = _num(info.get("freeCashflow"))
    shares = _num(info.get("sharesOutstanding"))

    def field(v):
        return EvidenceField(value=v, confidence=_YF_CONFIDENCE if v is not None else 0.0)

    snap = FinancialSnapshot(
        revenue_ttm=field(revenue),
        ebitda_ttm=field(ebitda),
        gross_margin_ttm=field(gross_margin),
        cash=field(cash),
        total_debt=field(total_debt),
        net_debt=field(net_debt),
        # nearest_debt_maturity is not in yfinance — left for the LLM/filing.
        fcf_ttm=field(fcf),
        shares_out=field(shares),
    )
    filled = {
        name for name in FinancialSnapshot.model_fields
        if getattr(snap, name).value is not None
    }

    # Context for valuation + cycle scoring (kept in raw_data).
    yf_data = {
        "price_52w_high": _num(info.get("fiftyTwoWeekHigh")),
        "price_52w_low": _num(info.get("fiftyTwoWeekLow")),
        "current_price": _num(info.get("currentPrice") or info.get("regularMarketPrice")),
        "market_cap": _num(info.get("marketCap")),
        "pe_ratio": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "price_to_book": _num(info.get("priceToBook")),
        "enterprise_value": _num(info.get("enterpriseValue")),
        "ebitda": ebitda,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "long_business_summary": (info.get("longBusinessSummary") or "")[:600],
    }
    return snap, yf_data, filled


def _is_sparse(snap: FinancialSnapshot, ticker: str) -> bool:
    """True if we should fall back to LLM filing extraction."""
    if not ticker or ticker.startswith("CIK"):
        return True  # placeholder ticker -> no yfinance coverage at all
    return any(getattr(snap, name).value is None for name in _REQUIRED)


def _financial_rows(candidate_id: str, snap: FinancialSnapshot, provenance_by_field: Dict[str, str]) -> List[dict]:
    """Evidence rows for a FinancialSnapshot with per-field provenance."""
    rows = []
    for name in FinancialSnapshot.model_fields:
        ev: EvidenceField = getattr(snap, name)
        if ev.value is None and ev.confidence == 0.0:
            continue  # nothing learned about this field
        eid = hashlib.sha256(
            f"{candidate_id}:financial:{name}".encode("utf-8")
        ).hexdigest()[:16]
        rows.append({
            "evidence_id": eid,
            "kind": "financial",
            "field": name,
            "value_json": json.dumps(ev.value, default=str),
            "confidence": ev.confidence,
            "source_id": ev.source_id,
            "provenance": provenance_by_field.get(name, "yfinance"),
            "snippet": ev.snippet,
            "extractor_version": ev.extractor_version,
        })
    return rows


async def _extract_one(
    client, config: Config, semaphore: asyncio.Semaphore, candidate: Candidate
) -> Tuple[Candidate, List[str]]:
    """Extract financials (hybrid) + moat for one candidate."""
    errors: List[str] = []
    ticker = candidate.get("ticker", "")
    cid = candidate["candidate_id"]

    # --- 1. yfinance fundamentals ---
    snap = FinancialSnapshot()
    provenance: Dict[str, str] = {}
    if _YF_AVAILABLE and ticker and not ticker.startswith("CIK"):
        async with semaphore:
            try:
                snap, yf_data, filled = await asyncio.to_thread(_yf_financials, ticker)
                raw = candidate.get("raw_data") or {}
                raw["yf_data"] = yf_data
                candidate["raw_data"] = raw
                for name in filled:
                    provenance[name] = "yfinance"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"extract: {ticker} yfinance failed: {exc}")

    # --- 2. LLM filing fallback on sparsity ---
    if _is_sparse(snap, ticker):
        prompt = load_prompt("financial_extractor").replace(
            "{candidate_block}", render_candidate_block(candidate)
        )
        async with semaphore:
            try:
                llm_snap: FinancialSnapshot = await extract_model(
                    client, config, prompt, FinancialSnapshot,
                    what=f"financials[{ticker}]",
                )
                guard_snippets(llm_snap, source_texts_of(candidate))
                # Fill only the fields yfinance left empty.
                for name in FinancialSnapshot.model_fields:
                    if getattr(snap, name).value is None:
                        llm_ev: EvidenceField = getattr(llm_snap, name)
                        if llm_ev.value is not None and llm_ev.confidence >= 0.5:
                            setattr(snap, name, llm_ev)
                            provenance[name] = "llm_filing"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"extract: {ticker} financial LLM fallback failed: {exc}")

    candidate["financial_snapshot"] = snap.model_dump()
    for row in _financial_rows(cid, snap, provenance):
        try:
            upsert_evidence(config.db_path, cid, row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"extract: {ticker} financial persist failed: {exc}")

    # --- 3. Moat extraction (LLM) ---
    prompt = load_prompt("moat_extractor").replace(
        "{candidate_block}", render_candidate_block(candidate)
    )
    async with semaphore:
        try:
            moat: MoatProxy = await extract_model(
                client, config, prompt, MoatProxy, what=f"moat[{ticker}]",
            )
            guard_snippets(moat, source_texts_of(candidate))
            candidate["moat_proxy"] = moat.model_dump()
            for row in evidence_rows(cid, "moat_proxy", moat, provenance="llm_filing"):
                upsert_evidence(config.db_path, cid, row)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"extract: {ticker} moat extraction failed: {exc}")

    return candidate, errors


async def extract_candidates(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    """Extract evidence for a batch. Returns (enriched, errors)."""
    if not candidates:
        return [], []

    client = get_client(config)
    semaphore = asyncio.Semaphore(config.max_concurrent_llm)
    results = await asyncio.gather(
        *(_extract_one(client, config, semaphore, c) for c in candidates)
    )

    enriched: List[Candidate] = []
    errors: List[str] = []
    for candidate, errs in results:
        errors.extend(errs)
        try:
            upsert_candidate(config.db_path, candidate)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"extract: db upsert failed for {candidate.get('ticker')}: {exc}")
        enriched.append(candidate)

    logger.info("extract_evidence: enriched %d candidates", len(enriched))
    return enriched, errors


def make_extract_evidence_node(config: Config):
    """Factory: returns an async graph node that closes over ``config``."""

    async def _node(state: dict) -> dict:
        enriched, errors = await extract_candidates(
            config, state.get("classified_candidates", [])
        )
        return {"enriched_candidates": enriched, "errors": errors}

    return _node
