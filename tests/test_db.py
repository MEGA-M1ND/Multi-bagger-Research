"""Smoke tests for the persistence layer and core state helpers."""

import tempfile
from pathlib import Path

import pytest

import sqlite3

from sndk_detector.db import (
    get_evidence,
    get_recent_candidate_ids,
    get_recent_scored_ids,
    get_watchlist,
    has_been_alerted,
    init_db,
    mark_as_alerted,
    upsert_candidate,
    upsert_evidence,
    upsert_scorecard,
    upsert_source_document,
)
from sndk_detector.state import BlueprintScore, make_candidate_id, new_candidate


@pytest.fixture()
def db_path():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "test.db")


def test_candidate_id_is_stable_and_dedups():
    a = make_candidate_id("ACME", "sec_edgar")
    b = make_candidate_id(" acme ", "SEC_EDGAR")  # whitespace/case insensitive
    assert a == b
    assert make_candidate_id("ACME", "news") != a


def test_init_db_is_idempotent(db_path):
    init_db(db_path)
    init_db(db_path)  # second call must not raise
    assert Path(db_path).exists()


def test_upsert_and_dedup_only_counts_scored(db_path):
    init_db(db_path)
    cand = new_candidate("ACME", "Acme Corp", "US", "sec_edgar")

    # Unscored candidates are not "recent" for dedup purposes.
    upsert_candidate(db_path, cand)
    assert get_recent_candidate_ids(db_path, 7) == set()

    # Once scored, it shows up in the recent set.
    cand["blueprint"] = BlueprintScore(
        structural_event=True,
        cyclical_trough=False,
        secular_tailwind=True,
        supply_constraint=True,
        undervalued_narrative=True,
        domain_edge=True,
        total_score=5,
        reasoning="t",
    )
    upsert_candidate(db_path, cand)
    assert get_recent_candidate_ids(db_path, 7) == {cand["candidate_id"]}


def test_alert_idempotency(db_path):
    init_db(db_path)
    cand = new_candidate("ACME", "Acme Corp", "US", "sec_edgar")
    cand["blueprint"] = BlueprintScore(
        structural_event=True,
        cyclical_trough=False,
        secular_tailwind=True,
        supply_constraint=True,
        undervalued_narrative=True,
        domain_edge=True,
        total_score=5,
        reasoning="t",
    )
    upsert_candidate(db_path, cand)

    assert has_been_alerted(db_path, cand["candidate_id"]) is False
    mark_as_alerted(db_path, cand, "hello")
    assert has_been_alerted(db_path, cand["candidate_id"]) is True


# --------------------------------------------------------------------------
# v2 schema: migration + new tables
# --------------------------------------------------------------------------

def test_migration_adds_v2_columns_to_existing_v1_db(db_path):
    # Simulate a v1 candidates table WITHOUT the v2 columns.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, company_name TEXT,
                market TEXT, source TEXT, price REAL, market_cap REAL, total_score INTEGER,
                blueprint_json TEXT, thesis TEXT, raw_data_json TEXT,
                first_seen TEXT NOT NULL, last_scored TEXT, updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    # init_db must migrate it in place without error.
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(candidates)")}
    finally:
        conn.close()
    assert {"cik", "event_family", "status", "priority_for_me"} <= cols


def test_v2_evidence_scorecard_watchlist_roundtrip(db_path):
    init_db(db_path)
    cand = new_candidate("ANEW", "Acme NewCo", "US", "sec_edgar")
    cand["cik"] = "0001999999"
    cand["scorecard"] = {"total_score": 72, "tier": "deep_dive"}
    upsert_candidate(db_path, cand)
    cid = cand["candidate_id"]

    upsert_source_document(db_path, cid, {
        "source_id": "doc1", "cik": "0001999999", "form": "10-12B",
        "accession": "a-1", "filename": "f.htm", "url": "https://x", "file_date": "2026-05-01",
        "fetched_text": "intends to separate",
    })
    upsert_evidence(db_path, cid, {
        "evidence_id": "ev1", "kind": "financial", "field": "net_debt",
        "value_json": "-100000000", "confidence": 0.85, "source_id": None,
        "provenance": "yfinance", "snippet": None, "extractor_version": "v1",
    })
    upsert_scorecard(db_path, cid, {
        "total_score": 72, "event_quality": 18, "tier": "deep_dive",
        "hard_fail": False, "hard_fail_reasons": [], "scorer_version": "v1",
    })

    assert len(get_evidence(db_path, cid, "financial")) == 1
    assert get_recent_scored_ids(db_path, 7) == {cid}
    wl = get_watchlist(db_path, "watchlist")
    assert len(wl) == 1 and wl[0]["ticker"] == "ANEW" and wl[0]["tier"] == "deep_dive"
