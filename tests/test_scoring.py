"""Tests for the deterministic v2 scoring, hard-fail gate, and snippet guard."""

from sndk_detector.scoring import compute_priority, evaluate_hard_fails, score
from sndk_detector.schemas import EvidenceField, EventSignal
from sndk_detector.valuation import compute_valuation_gap


def _E(value, conf=0.9):
    return {"value": value, "confidence": conf}


# --------------------------------------------------------------------------
# Hard fails
# --------------------------------------------------------------------------

def test_clean_company_no_hard_fail():
    fin = {k: _E(v) for k, v in {
        "fcf_ttm": 5e8, "cash": 1e9, "net_debt": 2e8, "ebitda_ttm": 8e8,
    }.items()}
    hard_fail, reasons = evaluate_hard_fails(fin, None, None)
    assert hard_fail is False and reasons == []


def test_cash_burn_triggers_hard_fail():
    fin = {"fcf_ttm": _E(-3e8), "cash": _E(1e8)}
    hard_fail, reasons = evaluate_hard_fails(fin, None, None)
    assert hard_fail is True
    assert any("runway" in r for r in reasons)


def test_low_confidence_financials_ignored():
    fin = {"fcf_ttm": _E(-3e8, 0.3), "cash": _E(1e8, 0.3)}
    hard_fail, _ = evaluate_hard_fails(fin, None, None)
    assert hard_fail is False


def test_high_conf_risk_flag_fails_low_conf_does_not():
    clean = {"net_debt": _E(-1e8)}
    assert evaluate_hard_fails(clean, None, {"governance_red_flag": _E(True, 0.9)})[0] is True
    assert evaluate_hard_fails(clean, None, {"governance_red_flag": _E(True, 0.55)})[0] is False


# --------------------------------------------------------------------------
# Weighted score
# --------------------------------------------------------------------------

def _strong_candidate():
    return {
        "candidate_id": "c1", "ticker": "ANEW",
        "raw_data": {"yf_data": {
            "current_price": 11.0, "price_52w_high": 40.0, "price_52w_low": 10.0,
            "long_business_summary": "Leading supplier of AI infrastructure and data center accelerators.",
            "industry": "Semiconductors",
        }},
        "event_signal": {"record_date": _E("2026-06-30"), "rationale": _E("unlock value"),
                         "spun_entity": _E("Acme NewCo")},
        "moat_proxy": {"market_position": _E("sole-source supplier"), "switching_costs": _E(True)},
        "financial_snapshot": {"net_debt": _E(-1e8), "fcf_ttm": _E(2e8), "nearest_debt_maturity": _E("2031")},
        "valuation_gap": {"gap_pct": _E(50.0)},
        "risk_flags": {"hard_fail": False, "reasons": []},
    }


def test_subscores_sum_to_total_capped():
    card = score(_strong_candidate(), thresholds=(40, 60, 75))
    raw = (card.event_quality + card.cycle_position + card.secular_tailwind
           + card.moat_proxies + card.valuation_dislocation + card.survivability)
    assert card.total_score == min(100, raw)
    assert card.tier in ("deep_dive", "starter")


def test_hard_fail_forces_reject_even_at_high_score():
    cand = _strong_candidate()
    cand["risk_flags"] = {"hard_fail": True, "reasons": ["dilution"]}
    card = score(cand, thresholds=(40, 60, 75))
    assert card.tier == "reject"
    assert card.total_score > 0  # score still computed for observability


def test_named_tailwind_required():
    cand = _strong_candidate()
    cand["raw_data"]["yf_data"]["long_business_summary"] = "We sell household goods."
    cand["raw_data"]["yf_data"]["industry"] = "Packaged Foods"
    cand["moat_proxy"] = {}
    assert score(cand, (40, 60, 75)).secular_tailwind == 0


def test_valuation_zero_when_no_gap():
    cand = _strong_candidate()
    cand["valuation_gap"] = {"gap_pct": {"value": None, "confidence": 0.0}}
    assert score(cand, (40, 60, 75)).valuation_dislocation == 0


def test_priority_overlay_from_domain_terms():
    assert compute_priority(_strong_candidate()) > 0


# --------------------------------------------------------------------------
# Valuation gap (injected fetcher — no network)
# --------------------------------------------------------------------------

def test_valuation_gap_no_peers_is_honest_zero():
    cand = {"candidate_id": "x", "ticker": "X", "raw_data": {"yf_data": {"sector": "Nowhere"}}}
    gap = compute_valuation_gap(cand, fetcher=lambda t: (10.0, None))
    assert gap.gap_pct.value is None and gap.gap_pct.confidence == 0.0


def test_valuation_gap_computes_cheapness():
    cand = {"candidate_id": "y", "ticker": "ANEW", "raw_data": {"yf_data": {
        "sector": "Technology", "enterprise_value": 2.2e9, "ebitda": 4e8,
    }}}
    peers = {"MSFT": (12.0, None), "ORCL": (11.0, None), "IBM": (10.0, None)}
    gap = compute_valuation_gap(cand, fetcher=lambda t: peers.get(t, (None, None)))
    assert gap.multiple_kind == "EV/EBITDA"
    assert gap.subject_multiple.value == 5.5
    assert gap.gap_pct.value > 0  # cheaper than peers


# --------------------------------------------------------------------------
# Snippet anti-hallucination guard
# --------------------------------------------------------------------------

def test_guard_snippets_demotes_hallucinated_quotes():
    from sndk_detector.nodes._llm import guard_snippets
    sig = EventSignal(
        event_family="spinoff",
        rationale=EvidenceField(value="separate", confidence=0.95, source_id="doc1",
                                snippet="intends to separate its storage business"),
        distribution_ratio=EvidenceField(value="1:2", confidence=0.9, source_id="doc1",
                                          snippet="one share for every two"),  # not in source
    )
    sources = {"doc1": "Acme intends to separate its storage business into a new company."}
    demoted = guard_snippets(sig, sources)
    assert demoted == 1
    assert sig.rationale.confidence == 0.95
    assert sig.distribution_ratio.confidence == 0.0
