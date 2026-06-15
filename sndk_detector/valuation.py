"""Peer-relative valuation for spinoff candidates.

Valuation dislocation is the single biggest scoring weight (20 pts), so it must
be computed mechanically — never narrated by the LLM. We compare the subject's
multiple against a hand-curated peer set's median.

Spinoff data is thin (fresh entities, no history), so the honest default is
**zero credit when we can't compute a real gap** — we never fabricate a positive
dislocation. The peer set is hand-curated rather than screener-derived because
the slice only needs to cover the sectors spinoffs actually cluster in.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Tuple

from .schemas import EvidenceField, ValuationGap
from .state import Candidate

logger = logging.getLogger(__name__)

# Hand-curated liquid peers by yfinance sector. Deliberately small + honest:
# when a candidate's sector isn't here, valuation simply scores 0 (logged).
PEER_MAP: Dict[str, List[str]] = {
    "Technology": ["MSFT", "ORCL", "IBM", "HPQ", "NTAP"],
    "Industrials": ["GE", "HON", "EMR", "ETN", "PH"],
    "Healthcare": ["JNJ", "ABT", "MDT", "BDX", "SYK"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "OXY"],
    "Basic Materials": ["LIN", "NEM", "FCX", "DOW", "NUE"],
    "Consumer Cyclical": ["HD", "NKE", "LOW", "TJX", "GPC"],
    "Consumer Defensive": ["PG", "KO", "PEP", "MDLZ", "KMB"],
    "Communication Services": ["GOOGL", "VZ", "T", "CMCSA", "OMC"],
    "Financial Services": ["JPM", "BAC", "SCHW", "USB", "PNC"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "EXC"],
    "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "O"],
}


def _num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _yf_multiple_fetcher(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """Fetch (ev_ebitda, price_to_book) for a peer ticker via yfinance.

    Network call, defensive: any failure returns (None, None).
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        ev = _num(info.get("enterpriseValue"))
        ebitda = _num(info.get("ebitda"))
        ev_ebitda = (ev / ebitda) if (ev and ebitda and ebitda > 0) else None
        pb = _num(info.get("priceToBook"))
        return ev_ebitda, pb
    except Exception:  # noqa: BLE001
        return None, None


def _empty_gap(reason: str) -> ValuationGap:
    """A gap with no usable signal — scorer will award 0. Records why."""
    return ValuationGap(
        peer_set=EvidenceField(value=reason, confidence=0.0, extractor_version="v1"),
    )


def compute_valuation_gap(
    candidate: Candidate,
    fetcher: Callable[[str], Tuple[Optional[float], Optional[float]]] = _yf_multiple_fetcher,
) -> ValuationGap:
    """Compute the subject-vs-peer multiple gap. Deterministic, honest about gaps.

    gap_pct > 0 means the subject trades CHEAPER than peers (dislocation upside).
    Returns a zero-confidence gap when peers or the subject multiple are
    unavailable — never a fabricated positive.
    """
    raw = candidate.get("raw_data") or {}
    yf_data = raw.get("yf_data") or {}
    sector = yf_data.get("sector")
    ticker = (candidate.get("ticker") or "").upper()

    peers = [p for p in PEER_MAP.get(sector or "", []) if p != ticker]
    if not peers:
        logger.debug("valuation: no peer set for sector=%r", sector)
        return _empty_gap(f"no peer set for sector {sector!r}")

    # --- subject multiple: prefer EV/EBITDA, fall back to P/B ---
    fin = candidate.get("financial_snapshot") or {}
    market_cap = _num(yf_data.get("market_cap"))
    ev = _num(yf_data.get("enterprise_value"))
    ebitda = _num(yf_data.get("ebitda")) or _num((fin.get("ebitda_ttm") or {}).get("value"))
    net_debt = _num((fin.get("net_debt") or {}).get("value"))
    if ev is None and market_cap is not None and net_debt is not None:
        ev = market_cap + net_debt

    subject_mult: Optional[float] = None
    kind = "EV/EBITDA"
    if ev is not None and ebitda is not None and ebitda > 0:
        subject_mult = ev / ebitda
    else:
        pb = _num(yf_data.get("price_to_book"))
        if pb is not None and pb > 0:
            subject_mult = pb
            kind = "P/B"

    if subject_mult is None:
        return _empty_gap("subject multiple uncomputable (no EV/EBITDA or P/B)")

    # --- peer median of the SAME multiple kind ---
    peer_vals: List[float] = []
    for p in peers:
        ev_ebitda, pb = fetcher(p)
        val = ev_ebitda if kind == "EV/EBITDA" else pb
        if val is not None and val > 0:
            peer_vals.append(val)

    if not peer_vals:
        return _empty_gap(f"no peer multiples available ({kind})")

    peer_vals.sort()
    n = len(peer_vals)
    peer_median = (
        peer_vals[n // 2] if n % 2 else (peer_vals[n // 2 - 1] + peer_vals[n // 2]) / 2
    )
    gap_pct = (peer_median - subject_mult) / peer_median * 100.0

    # Confidence = peer coverage fraction (how many of the curated peers we got),
    # capped — thin coverage means a less trustworthy gap.
    coverage = len(peer_vals) / len(peers)
    confidence = round(min(0.85, 0.4 + 0.5 * coverage), 2)

    return ValuationGap(
        subject_multiple=EvidenceField(value=round(subject_mult, 2), confidence=confidence),
        peer_median_multiple=EvidenceField(value=round(peer_median, 2), confidence=confidence),
        gap_pct=EvidenceField(value=round(gap_pct, 1), confidence=confidence),
        peer_set=EvidenceField(value=peers, confidence=confidence),
        multiple_kind=kind,
    )
