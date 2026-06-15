"""Deterministic scoring + hard-fail logic. Pure Python, no I/O.

This module is where v2 keeps the LLM out of the numbers. ``evaluate_hard_fails``
and ``score`` (added in the scoring step) read structured evidence and produce a
verdict mechanically, so a compelling narrative can never buy points it can't back
with data.

Evidence is read from the model_dump() of the pydantic schemas, i.e. each field
looks like ``{"value": ..., "confidence": 0.0..1.0, "source_id": ..., "snippet": ...}``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .schemas import Scorecard

# Confidence floor for a piece of evidence to count at all.
_MIN_CONF = 0.5
# Evidence in [_MIN_CONF, _WEAK_CONF) counts at half weight ("weak" rung).
_WEAK_CONF = 0.7
# A disqualifying LLM risk flag needs high confidence — we don't reject a name
# on a shaky inference.
_RISK_CONF = 0.7


def ev(snapshot: Optional[dict], field: str) -> Tuple[Any, float]:
    """Return (value, confidence) for one evidence field, or (None, 0.0)."""
    if not snapshot:
        return None, 0.0
    f = snapshot.get(field) or {}
    return f.get("value"), float(f.get("confidence") or 0.0)


def _num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_hard_fails(
    financial_snapshot: Optional[dict],
    moat_proxy: Optional[dict],
    risk_flags: Optional[dict] = None,
) -> Tuple[bool, List[str]]:
    """Deterministic + LLM-flag disqualifier gate.

    Returns (hard_fail, reasons). A True here forces tier='reject' regardless of
    the weighted score — "a high score WITH a hard fail is still a reject".

    Deterministic checks (only on confident financial evidence):
      * negative FCF with under ~1 year of cash runway (dilution/insolvency risk)
      * negative EBITDA while carrying net debt (can't service debt)
      * extreme leverage: net debt > 8x EBITDA
    Plus any LLM RiskFlag that is active at high confidence.
    """
    reasons: List[str] = []

    fcf, fcf_c = ev(financial_snapshot, "fcf_ttm")
    cash, cash_c = ev(financial_snapshot, "cash")
    net_debt, nd_c = ev(financial_snapshot, "net_debt")
    ebitda, eb_c = ev(financial_snapshot, "ebitda_ttm")

    fcf_n = _num(fcf) if fcf_c >= _MIN_CONF else None
    cash_n = _num(cash) if cash_c >= _MIN_CONF else None
    net_debt_n = _num(net_debt) if nd_c >= _MIN_CONF else None
    ebitda_n = _num(ebitda) if eb_c >= _MIN_CONF else None

    if fcf_n is not None and fcf_n < 0 and cash_n is not None and abs(fcf_n) > 0:
        runway_years = cash_n / abs(fcf_n)
        if runway_years < 1.0:
            reasons.append(
                f"under 1y cash runway with negative FCF "
                f"(cash {cash_n:,.0f} / burn {abs(fcf_n):,.0f} ~ {runway_years:.1f}y)"
            )

    if ebitda_n is not None and ebitda_n <= 0 and net_debt_n is not None and net_debt_n > 0:
        reasons.append("negative EBITDA while carrying net debt")

    if (
        ebitda_n is not None and ebitda_n > 0
        and net_debt_n is not None and net_debt_n > 0
        and net_debt_n / ebitda_n > 8.0
    ):
        reasons.append(f"extreme leverage: net debt {net_debt_n / ebitda_n:.1f}x EBITDA")

    # LLM disqualifier flags (high-confidence only).
    _RISK_LABELS = {
        "refinancing_12mo": "refinancing risk within 12 months",
        "material_dilution": "material dilution risk (shelf/cash burn)",
        "customer_concentration_risk": "customer concentration without contractual durability",
        "governance_red_flag": "governance / promotional red flag",
        "no_catalyst_path": "no credible catalyst path to earnings",
    }
    if risk_flags:
        for key, label in _RISK_LABELS.items():
            value, conf = ev(risk_flags, key)
            if value and conf >= _RISK_CONF:
                reasons.append(label)

    return (len(reasons) > 0), reasons


# ---------------------------------------------------------------------------
# Weighted 100-point score (deterministic)
# ---------------------------------------------------------------------------

# Named secular themes — a tailwind must be NAMED to earn points (no generic
# "growth" credit). Matched case-insensitively against business text + snippets.
_TAILWIND_TERMS = (
    "artificial intelligence", "ai infrastructure", "data center", "datacenter",
    "semiconductor", "gpu", "accelerator", "small modular reactor", "nuclear",
    "smr", "defense", "quantum", "robotics", "autonomous", "cybersecurity",
    "rare earth", "critical mineral", "grid", "electrification",
)
# Domain edge — themes where the operator (AI infra/security) has personal edge.
_DOMAIN_EDGE_TERMS = (
    "ai infrastructure", "artificial intelligence", "gpu", "accelerator",
    "data center", "datacenter", "cybersecurity", "security", "semiconductor",
)


def _conf_weight(conf: float) -> float:
    """Full credit at >= _WEAK_CONF, half credit on the weak rung, else none."""
    if conf >= _WEAK_CONF:
        return 1.0
    if conf >= _MIN_CONF:
        return 0.5
    return 0.0


def _tailwind_text(candidate: dict) -> str:
    """Concatenate the text we scan for named secular themes."""
    raw = candidate.get("raw_data") or {}
    yf_data = raw.get("yf_data") or {}
    parts = [
        yf_data.get("long_business_summary") or "",
        yf_data.get("industry") or "",
        raw.get("headline") or "",
        raw.get("summary") or "",
    ]
    moat = candidate.get("moat_proxy") or {}
    mp = moat.get("market_position") or {}
    if mp.get("snippet"):
        parts.append(mp["snippet"])
    return " ".join(parts).lower()


def _score_event_quality(event_signal: Optional[dict]) -> int:
    pts = 0.0
    _, rd_c = ev(event_signal, "record_date")
    _, rat_c = ev(event_signal, "rationale")
    spun_v, spun_c = ev(event_signal, "spun_entity")
    if rd_c >= _MIN_CONF:
        pts += 8 * _conf_weight(rd_c)
    if rat_c >= _MIN_CONF:
        pts += 6 * _conf_weight(rat_c)
    if spun_v and spun_c >= _MIN_CONF:
        pts += 6 * _conf_weight(spun_c)
    return min(20, round(pts))


def _score_cycle(candidate: dict) -> int:
    """Price near its 52-week low scores higher (cyclical trough proxy)."""
    yf_data = (candidate.get("raw_data") or {}).get("yf_data") or {}
    price = yf_data.get("current_price")
    hi = yf_data.get("price_52w_high")
    lo = yf_data.get("price_52w_low")
    if not (price and hi and lo) or hi <= lo:
        return 0
    position = (price - lo) / (hi - lo)  # 0 = at low, 1 = at high
    if position <= 0.25:
        return 15
    if position <= 0.50:
        return 10
    if position <= 0.75:
        return 5
    return 0


def _score_tailwind(candidate: dict) -> int:
    """Require a NAMED secular theme; otherwise 0."""
    text = _tailwind_text(candidate)
    if not any(term in text for term in _TAILWIND_TERMS):
        return 0
    pts = 14  # a named tailwind is present
    moat = candidate.get("moat_proxy") or {}
    mp_v, mp_c = ev(moat, "market_position")
    if mp_v and mp_c >= _MIN_CONF:
        pts += 6 * _conf_weight(mp_c)
    return min(20, round(pts))


def _score_moat(moat_proxy: Optional[dict]) -> int:
    if not moat_proxy:
        return 0
    pts = 0.0
    for field in ("switching_costs", "customer_concentration",
                  "contractual_durability", "market_position", "capital_intensity"):
        value, conf = ev(moat_proxy, field)
        if value and conf >= _MIN_CONF:
            pts += 5 * _conf_weight(conf)
    return min(15, round(pts))


def _score_valuation(valuation_gap: Optional[dict]) -> int:
    value, conf = ev(valuation_gap, "gap_pct")
    if value is None or conf < _MIN_CONF:
        return 0  # honest zero — never fabricate dislocation
    gap = _num(value)
    if gap is None or gap <= 0:
        return 0  # at/above peer multiple => no dislocation
    return max(0, min(20, round(gap / 2.0 * _conf_weight(conf))))


def _score_survivability(financial_snapshot: Optional[dict]) -> int:
    if not financial_snapshot:
        return 0
    pts = 0
    net_debt, nd_c = ev(financial_snapshot, "net_debt")
    nd = _num(net_debt) if nd_c >= _MIN_CONF else None
    if nd is not None and nd <= 0:
        pts += 6  # net cash position
    maturity, mat_c = ev(financial_snapshot, "nearest_debt_maturity")
    # No near-term maturity flagged (or net cash) -> credit. Conservative:
    # only award when we have net cash OR an explicit far maturity.
    if (nd is not None and nd <= 0) or (maturity and mat_c >= _MIN_CONF):
        pts += 5
    fcf, fcf_c = ev(financial_snapshot, "fcf_ttm")
    fcf_n = _num(fcf) if fcf_c >= _MIN_CONF else None
    if fcf_n is not None and fcf_n > 0:
        pts += 4
    return min(15, pts)


def compute_priority(candidate: dict) -> float:
    """Personal-attention overlay (0..1). Ranking only — never gates scoring."""
    text = _tailwind_text(candidate)
    hits = sum(1 for term in _DOMAIN_EDGE_TERMS if term in text)
    return round(min(1.0, hits / 3.0), 2)


def _tier(total: int, hard_fail: bool, thresholds: Tuple[int, int, int]) -> str:
    if hard_fail:
        return "reject"
    watchlist, deep_dive, starter = thresholds
    if total >= starter:
        return "starter"
    if total >= deep_dive:
        return "deep_dive"
    if total >= watchlist:
        return "watchlist"
    return "reject"


def score(
    candidate: dict,
    thresholds: Tuple[int, int, int] = (40, 60, 75),
    scorer_version: str = "v1",
) -> Scorecard:
    """Produce the deterministic 100-point Scorecard for a candidate.

    Reads only structured evidence already attached to the candidate
    (event_signal, financial_snapshot, moat_proxy, valuation_gap, risk_flags,
    raw_data.yf_data). A hard fail forces tier='reject' even at a high score.
    """
    event_quality = _score_event_quality(candidate.get("event_signal"))
    cycle = _score_cycle(candidate)
    tailwind = _score_tailwind(candidate)
    moat = _score_moat(candidate.get("moat_proxy"))
    valuation = _score_valuation(candidate.get("valuation_gap"))
    survivability = _score_survivability(candidate.get("financial_snapshot"))
    total = event_quality + cycle + tailwind + moat + valuation + survivability

    rf = candidate.get("risk_flags") or {}
    hard_fail = bool(rf.get("hard_fail"))
    reasons = list(rf.get("reasons") or [])

    return Scorecard(
        event_quality=event_quality,
        cycle_position=cycle,
        secular_tailwind=tailwind,
        moat_proxies=moat,
        valuation_dislocation=valuation,
        survivability=survivability,
        total_score=min(100, total),
        hard_fail=hard_fail,
        hard_fail_reasons=reasons,
        tier=_tier(min(100, total), hard_fail, thresholds),
    )
