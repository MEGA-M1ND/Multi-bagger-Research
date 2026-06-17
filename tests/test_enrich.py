"""Parser/mapping tests for the Perplexity enrichment node.

IMPORTANT: like the SEC tests, the JSON samples below are *documented-shape*
fixtures, NOT captured from a live call — api.perplexity.ai is blocked by this
environment's egress allowlist. If you can reach Perplexity and the live
``finance_search`` / chat response shape differs, replace these fixtures and
adjust ``_parse_financials`` / ``_apply_financials`` / ``_extract_citations``.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sndk_detector.db import init_db, upsert_candidate
from sndk_detector.nodes.enrich import (
    _apply_financials,
    _as_float,
    _dedup_for_enrich,
    _extract_citations,
    _parse_financials,
)
from sndk_detector.state import new_candidate


class _FakeResp:
    """Stand-in for an OpenAI/Perplexity response exposing model_dump()."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_as_float_coerces_and_guards():
    assert _as_float("12.5") == 12.5
    assert _as_float(7) == 7.0
    assert _as_float(None) is None
    assert _as_float("n/a") is None


def test_parse_financials_handles_garbage():
    assert _parse_financials(None) == {}
    assert _parse_financials("not json") == {}
    assert _parse_financials("[1, 2, 3]") == {}  # not a dict
    assert _parse_financials('{"price": 10}') == {"price": 10}


def test_apply_financials_maps_numbers_not_prose():
    cand = new_candidate("ACME", "Acme Corp", "US", "sec_edgar")
    data = {
        "price": "42.5",
        "currency": "USD",
        "market_cap": 1_000_000,
        "fundamentals": {"pe_ratio": 11.2, "margin_trend": "near multi-year low"},
        "corporate_events": ["Mar 2025: announced spinoff"],
    }
    _apply_financials(cand, data, citations=["http://example.com/q"])

    assert cand["price"] == 42.5
    assert cand["market_cap"] == 1_000_000.0
    assert cand["raw_data"]["fundamentals"]["pe_ratio"] == 11.2
    assert cand["raw_data"]["corporate_events"] == ["Mar 2025: announced spinoff"]
    assert cand["raw_data"]["currency"] == "USD"
    assert cand["raw_data"]["financial_citations"] == ["http://example.com/q"]


def test_apply_financials_skips_missing_fields():
    cand = new_candidate("ACME", "Acme Corp", "US", "sec_edgar")
    _apply_financials(cand, {"price": None, "fundamentals": {}}, citations=[])
    # Nothing usable -> price untouched (stays None) and no fundamentals attached.
    assert cand.get("price") is None
    assert "fundamentals" not in cand["raw_data"]


def test_extract_citations_both_shapes():
    assert _extract_citations(_FakeResp({"citations": ["a", "b"]})) == ["a", "b"]
    sr = _FakeResp({"search_results": [{"url": "u1"}, {"title": "t2"}]})
    assert _extract_citations(sr) == ["u1", "t2"]
    assert _extract_citations(_FakeResp({})) == []


def test_dedup_for_enrich_skips_recent_and_dups():
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "t.db")
        init_db(db_path)
        config = SimpleNamespace(db_path=db_path, research_lookback_days=7)

        fresh = new_candidate("FRESH", "Fresh Co", "US", "sec_edgar")
        already = new_candidate("OLD", "Old Co", "US", "sec_edgar")
        already["raw_data"] = {"research": {"summary": "x", "citations": []}}
        upsert_candidate(db_path, already)  # stamps last_enriched

        # `fresh` twice (intra-run dup) + `already` (recently enriched).
        to_enrich, skipped = _dedup_for_enrich(config, [fresh, fresh, already])
        assert [c["ticker"] for c in to_enrich] == ["FRESH"]
        assert skipped == 2
