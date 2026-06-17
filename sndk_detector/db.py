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
from typing import Iterator, Optional, Set

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
                last_enriched  TEXT,
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

            CREATE INDEX IF NOT EXISTS idx_candidates_last_scored
                ON candidates(last_scored);
            CREATE INDEX IF NOT EXISTS idx_alerts_candidate
                ON alerts(candidate_id);
            """
        )
        # Idempotent migration: add last_enriched to pre-existing databases
        # created before the enrichment stage was introduced.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(candidates)")}
        if "last_enriched" not in cols:
            conn.execute("ALTER TABLE candidates ADD COLUMN last_enriched TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_last_enriched "
            "ON candidates(last_enriched)"
        )


def upsert_candidate(db_path: str, candidate: Candidate) -> None:
    """Insert or update a candidate, keyed by candidate_id.

    ``first_seen`` is preserved across updates; ``last_scored`` is only stamped
    when the candidate carries a blueprint (i.e. it was scored this run), and
    ``last_enriched`` only when it carries enrichment data (fundamentals or
    research from the Perplexity enrich stage).
    """
    blueprint = candidate.get("blueprint")
    total_score = blueprint["total_score"] if blueprint else None
    scored_now = _utcnow_iso() if blueprint else None

    raw_data = candidate.get("raw_data") or {}
    enriched = bool(raw_data.get("fundamentals") or raw_data.get("research"))
    enriched_now = _utcnow_iso() if enriched else None

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO candidates (
                candidate_id, ticker, company_name, market, source,
                price, market_cap, total_score, blueprint_json, thesis,
                raw_data_json, first_seen, last_scored, last_enriched, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                ticker        = excluded.ticker,
                company_name  = excluded.company_name,
                market        = excluded.market,
                source        = excluded.source,
                price         = COALESCE(excluded.price, candidates.price),
                market_cap    = COALESCE(excluded.market_cap, candidates.market_cap),
                -- Only overwrite score fields when we have a fresh score.
                total_score    = COALESCE(excluded.total_score, candidates.total_score),
                blueprint_json = COALESCE(excluded.blueprint_json, candidates.blueprint_json),
                thesis         = COALESCE(excluded.thesis, candidates.thesis),
                raw_data_json  = excluded.raw_data_json,
                last_scored    = COALESCE(excluded.last_scored, candidates.last_scored),
                last_enriched  = COALESCE(excluded.last_enriched, candidates.last_enriched),
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
                json.dumps(raw_data),
                _utcnow_iso(),
                scored_now,
                enriched_now,
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


def get_recently_enriched_ids(db_path: str, lookback_days: int) -> Set[str]:
    """Return candidate_ids enriched within the lookback window.

    Used to skip re-enriching (and thus avoid burning Perplexity calls) on
    candidates we've already grounded recently. Only counts rows that were
    actually enriched (last_enriched IS NOT NULL).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT candidate_id FROM candidates
            WHERE last_enriched IS NOT NULL AND last_enriched >= ?
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
    blueprint = candidate.get("blueprint")
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO alerts (candidate_id, ticker, total_score, message, sent_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                candidate["candidate_id"],
                candidate.get("ticker"),
                blueprint["total_score"] if blueprint else None,
                message,
                _utcnow_iso(),
            ),
        )
