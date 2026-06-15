"""SQLite persistence layer (stdlib sqlite3, no ORM).

Two tables:
  * candidates — every candidate we've ever seen, with the latest score/thesis.
  * alerts     — an append-only log of what was sent and when.

``candidate_id`` (stable hash of ticker+source) is the idempotency key
everywhere: upserts key off it, dedup reads off it, and alert-suppression
checks off it.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Set

from .state import Candidate


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Connection context manager with sane defaults and dict-like rows."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # WAL improves concurrent read/write behaviour; harmless for single-user.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict) -> None:
    """Idempotently ADD COLUMN any of ``columns`` (name -> type) that are missing.

    SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we inspect PRAGMA table_info
    first. Keeps init_db safe to call on every startup even as the schema grows.
    """
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, coltype in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")


def init_db(db_path: str) -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id   TEXT PRIMARY KEY,
                ticker         TEXT NOT NULL,
                company_name   TEXT,
                market         TEXT,
                source         TEXT,
                price          REAL,
                market_cap     REAL,
                total_score    INTEGER,
                blueprint_json TEXT,
                thesis         TEXT,
                raw_data_json  TEXT,
                first_seen     TEXT NOT NULL,
                last_scored    TEXT,
                updated_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                ticker       TEXT,
                total_score  INTEGER,
                message      TEXT,
                sent_at      TEXT NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
            );

            -- v2: provenance anchor. One candidate can have many source docs
            -- (the spinoff filing + its 10-K/10-Q/424B for fundamentals).
            CREATE TABLE IF NOT EXISTS source_documents (
                source_id     TEXT PRIMARY KEY,
                candidate_id  TEXT NOT NULL,
                cik           TEXT,
                form          TEXT,
                accession     TEXT,
                filename      TEXT,
                url           TEXT,
                file_date     TEXT,
                fetched_text  TEXT,
                first_seen    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
            );

            -- v2: one polymorphic table for every kind of evidence claim, so the
            -- audit contract (value+confidence+source+snippet+provenance) is uniform.
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id       TEXT PRIMARY KEY,
                candidate_id      TEXT NOT NULL,
                kind              TEXT NOT NULL,
                field             TEXT NOT NULL,
                value_json        TEXT,
                confidence        REAL,
                source_id         TEXT,
                provenance        TEXT NOT NULL,
                snippet           TEXT,
                extractor_version TEXT NOT NULL,
                first_seen        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
            );

            -- v2: the 100-point weighted scorecard (one per candidate).
            CREATE TABLE IF NOT EXISTS scorecards (
                candidate_id          TEXT PRIMARY KEY,
                total_score           INTEGER,
                subscores_json        TEXT,
                hard_fail             INTEGER NOT NULL,
                hard_fail_reasons_json TEXT,
                tier                  TEXT,
                scored_at             TEXT NOT NULL,
                scorer_version        TEXT NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
            );

            -- v2: the memo + critic synthesis (one per candidate).
            CREATE TABLE IF NOT EXISTS memos (
                candidate_id  TEXT PRIMARY KEY,
                memo_json     TEXT,
                critic_json   TEXT,
                memo_path     TEXT,
                generated_at  TEXT NOT NULL,
                memo_version  TEXT NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_last_scored
                ON candidates(last_scored);
            CREATE INDEX IF NOT EXISTS idx_alerts_candidate
                ON alerts(candidate_id);
            CREATE INDEX IF NOT EXISTS idx_srcdoc_candidate
                ON source_documents(candidate_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_candidate
                ON evidence(candidate_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_kind
                ON evidence(candidate_id, kind);
            CREATE INDEX IF NOT EXISTS idx_scorecards_scored_at
                ON scorecards(scored_at);
            """
        )
        # v2 columns on the existing candidates table (idempotent migration).
        _ensure_columns(
            conn,
            "candidates",
            {
                "cik": "TEXT",
                "event_family": "TEXT",
                "status": "TEXT",
                "priority_for_me": "REAL",
            },
        )


def upsert_candidate(db_path: str, candidate: Candidate) -> None:
    """Insert or update a candidate, keyed by candidate_id.

    ``first_seen`` is preserved across updates; ``last_scored`` is only stamped
    when the candidate actually carries a score (scorecard in v2, blueprint in
    the legacy path). ``total_score`` now holds the 0-100 weighted score when a
    scorecard is present, falling back to the legacy 0-6 blueprint total.
    """
    scorecard = candidate.get("scorecard")
    blueprint = candidate.get("blueprint")
    if scorecard:
        total_score = scorecard.get("total_score")
    elif blueprint:
        total_score = blueprint["total_score"]
    else:
        total_score = None
    scored_now = _utcnow_iso() if (scorecard or blueprint) else None

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO candidates (
                candidate_id, ticker, company_name, market, source,
                price, market_cap, total_score, blueprint_json, thesis,
                raw_data_json, cik, event_family, status, priority_for_me,
                first_seen, last_scored, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                ticker        = excluded.ticker,
                company_name  = excluded.company_name,
                market        = excluded.market,
                source        = excluded.source,
                price         = excluded.price,
                market_cap    = excluded.market_cap,
                -- Only overwrite score fields when we have a fresh score.
                total_score    = COALESCE(excluded.total_score, candidates.total_score),
                blueprint_json = COALESCE(excluded.blueprint_json, candidates.blueprint_json),
                thesis         = COALESCE(excluded.thesis, candidates.thesis),
                raw_data_json  = excluded.raw_data_json,
                cik            = COALESCE(excluded.cik, candidates.cik),
                event_family   = COALESCE(excluded.event_family, candidates.event_family),
                status         = COALESCE(excluded.status, candidates.status),
                priority_for_me = COALESCE(excluded.priority_for_me, candidates.priority_for_me),
                last_scored    = COALESCE(excluded.last_scored, candidates.last_scored),
                updated_at     = excluded.updated_at
            """,
            (
                candidate["candidate_id"],
                candidate.get("ticker"),
                candidate.get("company_name"),
                candidate.get("market"),
                candidate.get("source"),
                candidate.get("price"),
                candidate.get("market_cap"),
                total_score,
                json.dumps(blueprint) if blueprint else None,
                candidate.get("thesis"),
                json.dumps(candidate.get("raw_data") or {}),
                candidate.get("cik"),
                candidate.get("event_family"),
                candidate.get("status"),
                candidate.get("priority_for_me"),
                _utcnow_iso(),
                scored_now,
                _utcnow_iso(),
            ),
        )


def get_recent_candidate_ids(db_path: str, lookback_days: int) -> Set[str]:
    """Return candidate_ids scored within the lookback window.

    Used to skip re-scoring (and thus avoid burning LLM tokens) on candidates
    we've already evaluated recently. Only counts rows that were actually
    scored (last_scored IS NOT NULL).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT candidate_id FROM candidates
            WHERE last_scored IS NOT NULL AND last_scored >= ?
            """,
            (cutoff,),
        ).fetchall()
    return {row["candidate_id"] for row in rows}


def has_been_alerted(db_path: str, candidate_id: str) -> bool:
    """True if we've ever sent an alert for this candidate_id."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM alerts WHERE candidate_id = ? LIMIT 1",
            (candidate_id,),
        ).fetchone()
    return row is not None


def mark_as_alerted(
    db_path: str,
    candidate: Candidate,
    message: str,
) -> None:
    """Record that an alert was sent for this candidate."""
    scorecard = candidate.get("scorecard")
    blueprint = candidate.get("blueprint")
    if scorecard:
        total_score = scorecard.get("total_score")
    elif blueprint:
        total_score = blueprint["total_score"]
    else:
        total_score = None
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO alerts (candidate_id, ticker, total_score, message, sent_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                candidate["candidate_id"],
                candidate.get("ticker"),
                total_score,
                message,
                _utcnow_iso(),
            ),
        )


# ---------------------------------------------------------------------------
# v2: source documents, evidence, scorecards, memos
# ---------------------------------------------------------------------------

def upsert_source_document(db_path: str, candidate_id: str, doc: dict) -> None:
    """Insert/update a source document. COALESCE on fetched_text so a later
    pass that re-discovers the doc without re-fetching text doesn't wipe it.

    ``doc`` keys: source_id, cik, form, accession, filename, url, file_date,
    fetched_text.
    """
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO source_documents (
                source_id, candidate_id, cik, form, accession, filename,
                url, file_date, fetched_text, first_seen, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                cik          = excluded.cik,
                form         = excluded.form,
                accession    = excluded.accession,
                filename     = excluded.filename,
                url          = excluded.url,
                file_date    = excluded.file_date,
                fetched_text = COALESCE(excluded.fetched_text, source_documents.fetched_text),
                updated_at   = excluded.updated_at
            """,
            (
                doc["source_id"],
                candidate_id,
                doc.get("cik"),
                doc.get("form"),
                doc.get("accession"),
                doc.get("filename"),
                doc.get("url"),
                doc.get("file_date"),
                doc.get("fetched_text"),
                _utcnow_iso(),
                _utcnow_iso(),
            ),
        )


def upsert_evidence(db_path: str, candidate_id: str, row: dict) -> None:
    """Insert/update one evidence row.

    ``evidence_id`` is the idempotency key, so re-extracting the same field with
    the same extractor_version overwrites cleanly. ``row`` keys: evidence_id,
    kind, field, value_json, confidence, source_id, provenance, snippet,
    extractor_version.
    """
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO evidence (
                evidence_id, candidate_id, kind, field, value_json, confidence,
                source_id, provenance, snippet, extractor_version,
                first_seen, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(evidence_id) DO UPDATE SET
                value_json        = excluded.value_json,
                confidence        = excluded.confidence,
                source_id         = excluded.source_id,
                provenance        = excluded.provenance,
                snippet           = excluded.snippet,
                extractor_version = excluded.extractor_version,
                updated_at        = excluded.updated_at
            """,
            (
                row["evidence_id"],
                candidate_id,
                row["kind"],
                row["field"],
                row.get("value_json"),
                row.get("confidence"),
                row.get("source_id"),
                row["provenance"],
                row.get("snippet"),
                row.get("extractor_version", "v1"),
                _utcnow_iso(),
                _utcnow_iso(),
            ),
        )


def get_evidence(
    db_path: str, candidate_id: str, kind: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return evidence rows for a candidate, optionally filtered by kind."""
    with _connect(db_path) as conn:
        if kind is None:
            rows = conn.execute(
                "SELECT * FROM evidence WHERE candidate_id = ?", (candidate_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM evidence WHERE candidate_id = ? AND kind = ?",
                (candidate_id, kind),
            ).fetchall()
    return [dict(row) for row in rows]


def upsert_scorecard(db_path: str, candidate_id: str, scorecard: dict) -> None:
    """Persist a scorecard. ``scorecard`` is a Scorecard.model_dump()."""
    subscores = {
        k: scorecard.get(k)
        for k in (
            "event_quality", "cycle_position", "secular_tailwind",
            "moat_proxies", "valuation_dislocation", "survivability",
        )
    }
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO scorecards (
                candidate_id, total_score, subscores_json, hard_fail,
                hard_fail_reasons_json, tier, scored_at, scorer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                total_score            = excluded.total_score,
                subscores_json         = excluded.subscores_json,
                hard_fail              = excluded.hard_fail,
                hard_fail_reasons_json = excluded.hard_fail_reasons_json,
                tier                   = excluded.tier,
                scored_at              = excluded.scored_at,
                scorer_version         = excluded.scorer_version
            """,
            (
                candidate_id,
                scorecard.get("total_score"),
                json.dumps(subscores),
                1 if scorecard.get("hard_fail") else 0,
                json.dumps(scorecard.get("hard_fail_reasons") or []),
                scorecard.get("tier"),
                _utcnow_iso(),
                scorecard.get("scorer_version", "v1"),
            ),
        )


def upsert_memo(db_path: str, candidate_id: str, memo: dict) -> None:
    """Persist a memo + critic. ``memo`` keys: memo_json, critic_json, memo_path."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO memos (
                candidate_id, memo_json, critic_json, memo_path,
                generated_at, memo_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                memo_json    = excluded.memo_json,
                critic_json  = excluded.critic_json,
                memo_path    = COALESCE(excluded.memo_path, memos.memo_path),
                generated_at = excluded.generated_at,
                memo_version = excluded.memo_version
            """,
            (
                candidate_id,
                memo.get("memo_json"),
                memo.get("critic_json"),
                memo.get("memo_path"),
                _utcnow_iso(),
                memo.get("memo_version", "v1"),
            ),
        )


def get_recent_scored_ids(db_path: str, lookback_days: int) -> Set[str]:
    """Return candidate_ids scored (scorecard written) within the window.

    The v2 analog of get_recent_candidate_ids — keyed on scorecards.scored_at so
    we skip re-running the (now more expensive) evidence+scoring chain on
    candidates evaluated recently.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT candidate_id FROM scorecards WHERE scored_at >= ?",
            (cutoff,),
        ).fetchall()
    return {row["candidate_id"] for row in rows}


_TIER_RANK = {"reject": 0, "watchlist": 1, "deep_dive": 2, "starter": 3}


def get_watchlist(db_path: str, min_tier: str = "watchlist") -> List[Dict[str, Any]]:
    """Return scored candidates at or above ``min_tier``, ranked for output.

    Joins candidates + scorecards; orders by total_score desc, then
    priority_for_me desc (the personal-attention overlay). Used by the weekly
    watchlist writer.
    """
    floor = _TIER_RANK.get(min_tier, 1)
    allowed = [t for t, r in _TIER_RANK.items() if r >= floor]
    placeholders = ",".join("?" for _ in allowed)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT c.candidate_id, c.ticker, c.company_name, c.market,
                   c.event_family, c.priority_for_me,
                   s.total_score, s.tier, s.subscores_json,
                   s.hard_fail, s.scored_at
            FROM scorecards s
            JOIN candidates c ON c.candidate_id = s.candidate_id
            WHERE s.tier IN ({placeholders})
            ORDER BY s.total_score DESC, c.priority_for_me DESC
            """,
            allowed,
        ).fetchall()
    return [dict(row) for row in rows]
