"""Parser tests for SEC EDGAR ingestion.

IMPORTANT: the sample below is a *documented-shape* fixture for the EDGAR
full-text search (efts.sec.gov) Elasticsearch response. It was NOT captured from
a live call — the SEC hosts are blocked by this environment's egress allowlist.
If you can reach SEC and the live shape differs, replace this fixture with a real
sample and adjust ``_extract_candidate`` accordingly.
"""

from sndk_detector.nodes.ingest_sec import (
    _extract_candidate,
    _parse_display_name,
    _warn_if_unexpected_shape,
)

# A representative EDGAR FTS response (documented shape).
SAMPLE_RESPONSE = {
    "hits": {
        "total": {"value": 2, "relation": "eq"},
        "hits": [
            {
                "_id": "0000320193-24-000123:form8k.htm",
                "_source": {
                    "display_names": ["Apple Inc. (AAPL) (CIK 0000320193)"],
                    "ciks": ["0000320193"],
                    "form": "8-K",
                    "root_forms": ["8-K"],
                    "file_date": "2024-05-01",
                    "file_type": "8-K",
                    "file_description": "Spin-off of subsidiary",
                    "biz_states": ["CA"],
                    "sics": ["3571"],
                },
            },
            {
                # A filer with no ticker in its display name — falls back to CIK.
                "_id": "0001999999-24-000001:forms1.htm",
                "_source": {
                    "display_names": ["Privately Held Spinco LLC (CIK 0001999999)"],
                    "ciks": ["0001999999"],
                    "form": "S-1",
                    "file_date": "2024-06-01",
                },
            },
        ],
    }
}


def test_parse_display_name_with_ticker():
    ticker, company = _parse_display_name("Apple Inc. (AAPL) (CIK 0000320193)")
    assert ticker == "AAPL"
    assert company == "Apple Inc."


def test_parse_display_name_without_ticker():
    ticker, company = _parse_display_name("Privately Held Spinco LLC (CIK 0001999999)")
    assert ticker is None
    assert company == "Privately Held Spinco LLC"


def test_extract_candidate_happy_path():
    hit = SAMPLE_RESPONSE["hits"]["hits"][0]
    cand = _extract_candidate(hit, '"spin-off"')
    assert cand is not None
    assert cand["ticker"] == "AAPL"
    assert cand["company_name"] == "Apple Inc."
    assert cand["market"] == "US"
    assert cand["source"] == "sec_edgar"
    assert cand["raw_data"]["form"] == "8-K"
    assert cand["raw_data"]["accession"] == "0000320193-24-000123"
    assert cand["raw_data"]["matched_query"] == '"spin-off"'


def test_extract_candidate_cik_fallback():
    hit = SAMPLE_RESPONSE["hits"]["hits"][1]
    cand = _extract_candidate(hit, '"spin-off"')
    assert cand is not None
    assert cand["ticker"] == "CIK0001999999"  # no ticker -> stable CIK identity


def test_extract_candidate_handles_garbage():
    assert _extract_candidate({}, "q") is None
    assert _extract_candidate({"_source": {}}, "q") is None


def test_shape_guard_flags_unexpected():
    errors = []
    _warn_if_unexpected_shape({"unexpected": True}, errors)
    assert errors and "unexpected SEC response shape" in errors[0]

    errors = []
    _warn_if_unexpected_shape(SAMPLE_RESPONSE, errors)
    assert errors == []
