"""Smoke tests for the persistence layer and core state helpers."""

import tempfile
from pathlib import Path

import pytest

from sndk_detector.db import (
    get_recent_candidate_ids,
    has_been_alerted,
    init_db,
    mark_as_alerted,
    upsert_candidate,
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
