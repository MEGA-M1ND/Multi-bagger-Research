"""output node: write deep-dive memos + a ranked watchlist to disk.

This is the operating model the v2 design calls for — a research triage engine
whose primary artifacts are files a human reads, not buy signals:
  * output/memos/<ticker>_<date>.md  for deep_dive/starter candidates
  * output/watchlist_<date>.md        ranked list of everything >= watchlist

Disk failures are recorded as errors, never crash the run.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

from ..config import Config
from ..db import get_watchlist, upsert_memo
from ..state import Candidate

logger = logging.getLogger(__name__)


def _run_date(state: dict) -> str:
    ts = state.get("run_timestamp") or ""
    return ts[:10] if len(ts) >= 10 else "undated"


def _evidence_table(candidate: Candidate) -> str:
    """A small markdown table of source documents (click-through provenance)."""
    docs = (candidate.get("raw_data") or {}).get("source_documents") or []
    if not docs:
        return "_No source documents._"
    lines = ["| Form | Date | Link |", "| --- | --- | --- |"]
    for d in docs:
        lines.append(f"| {d.get('form') or '?'} | {d.get('file_date') or '?'} | {d.get('url') or ''} |")
    return "\n".join(lines)


def _render_memo(candidate: Candidate) -> str:
    sc = candidate.get("scorecard") or {}
    memo = candidate.get("memo") or {}
    critic = candidate.get("critic") or {}
    rf = candidate.get("risk_flags") or {}

    sub = (
        f"event {sc.get('event_quality',0)} · cycle {sc.get('cycle_position',0)} · "
        f"tailwind {sc.get('secular_tailwind',0)} · moat {sc.get('moat_proxies',0)} · "
        f"valuation {sc.get('valuation_dislocation',0)} · survivability {sc.get('survivability',0)}"
    )
    lines = [
        f"# {candidate.get('ticker')} — {candidate.get('company_name')}",
        "",
        f"**Score: {sc.get('total_score', 0)}/100 · Tier: {sc.get('tier','?')} · "
        f"Event: {candidate.get('event_family','?')} · Priority: {candidate.get('priority_for_me',0)}**",
        "",
        f"Subscores: {sub}",
        "",
    ]
    if rf.get("hard_fail"):
        lines += [f"> ⚠️ HARD FAIL: {'; '.join(rf.get('reasons') or [])}", ""]

    lines += [
        "## Why now", memo.get("why_now", "unknown"), "",
        "## Why mispriced", memo.get("why_mispriced", "unknown"), "",
        "## What must happen (12m)", memo.get("what_must_happen_12m", "unknown"), "",
        "## Bear case", memo.get("bear_case", "unknown"), "",
    ]
    if memo.get("disconfirming_evidence"):
        lines += ["## Disconfirming evidence (kill conditions)"]
        lines += [f"- {x}" for x in memo["disconfirming_evidence"]] + [""]
    if memo.get("next_checkpoints"):
        lines += ["## Next checkpoints"]
        lines += [f"- {x}" for x in memo["next_checkpoints"]] + [""]
    if critic:
        lines += [
            "## Critic", "",
            f"**Short-seller view:** {critic.get('short_seller_view','unknown')}", "",
            f"**Most likely invalidating metric:** {critic.get('most_likely_invalidating_metric','unknown')}", "",
        ]
        if critic.get("unsupported_claims"):
            lines += ["**Unsupported claims:**"] + [f"- {x}" for x in critic["unsupported_claims"]] + [""]
        if critic.get("double_counted_factors"):
            lines += ["**Double-counted factors:**"] + [f"- {x}" for x in critic["double_counted_factors"]] + [""]

    lines += ["## Sources", _evidence_table(candidate), ""]
    return "\n".join(lines)


def _render_watchlist(rows: List[dict], run_date: str) -> str:
    lines = [
        f"# SNDK Watchlist — {run_date}",
        "",
        "| Rank | Ticker | Company | Score | Tier | Event | Priority |",
        "| ---: | --- | --- | ---: | --- | --- | ---: |",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r.get('ticker')} | {r.get('company_name') or ''} | "
            f"{r.get('total_score')} | {r.get('tier')} | {r.get('event_family') or ''} | "
            f"{r.get('priority_for_me') if r.get('priority_for_me') is not None else ''} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    config: Config, candidates: List[Candidate], run_date: str
) -> Tuple[List[str], List[str]]:
    """Write memos + watchlist. Returns (written_paths, errors)."""
    written: List[str] = []
    errors: List[str] = []

    memo_dir = os.path.join(config.output_dir, "memos")
    try:
        os.makedirs(memo_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"output: cannot create {memo_dir}: {exc}")
        return written, errors

    # Deep-dive memos for the top tiers.
    for cand in candidates:
        tier = (cand.get("scorecard") or {}).get("tier")
        if tier not in ("deep_dive", "starter") or not cand.get("memo"):
            continue
        path = os.path.join(memo_dir, f"{cand.get('ticker')}_{run_date}.md")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_render_memo(cand))
            written.append(path)
            upsert_memo(config.db_path, cand["candidate_id"], {
                "memo_json": json.dumps(cand.get("memo")),
                "critic_json": json.dumps(cand.get("critic")) if cand.get("critic") else None,
                "memo_path": path,
                "memo_version": config.scorer_version,
            })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"output: memo write failed for {cand.get('ticker')}: {exc}")

    # Ranked watchlist (everything >= watchlist, read back from the DB).
    try:
        rows = get_watchlist(config.db_path, "watchlist")
        wl_path = os.path.join(config.output_dir, f"watchlist_{run_date}.md")
        with open(wl_path, "w", encoding="utf-8") as fh:
            fh.write(_render_watchlist(rows, run_date))
        written.append(wl_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"output: watchlist write failed: {exc}")

    return written, errors


def make_output_node(config: Config):
    """Factory: returns a graph node that closes over ``config``."""

    def _node(state: dict) -> dict:
        written, errors = write_outputs(
            config, state.get("decided_candidates", []), _run_date(state)
        )
        if written:
            logger.info("output: wrote %d file(s)", len(written))
        return {"errors": errors}

    return _node
