"""Entry point: CLI, run orchestration, and the run summary.

Usage:
    sndk-detector --once                  # single run
    sndk-detector --schedule              # loop every 6h (default)
    sndk-detector --schedule --interval 3 # loop every 3h

Or without installing the console script:
    uv run python -m sndk_detector.main --once
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone

from .config import Config, ConfigError, load_config
from .db import init_db
from .graph import build_graph, initial_state
from .state import AgentState

logger = logging.getLogger("sndk_detector")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def run_once(config: Config, graph) -> AgentState:
    """Execute one full pass of the graph and return the final state.

    ``ainvoke`` (not ``invoke``) because the scoring node is async.
    """
    final_state: AgentState = await graph.ainvoke(initial_state())
    return final_state


def _print_summary(state: AgentState) -> None:
    """Print a clear, observable run summary (v2: tiered, 0-100)."""
    candidates = state.get("candidates", [])
    scored = state.get("decided_candidates", []) or state.get("scored_candidates", [])
    alerted = state.get("alert_queue", [])
    errors = state.get("errors", [])

    # Candidate counts per source, for at-a-glance source health.
    by_source: dict[str, int] = {}
    for cand in candidates:
        by_source[cand.get("source", "?")] = by_source.get(cand.get("source", "?"), 0) + 1

    # Tier + hard-fail counts.
    tiers = {"starter": 0, "deep_dive": 0, "watchlist": 0, "reject": 0}
    hard_fails = 0
    for cand in scored:
        sc = cand.get("scorecard") or {}
        tiers[sc.get("tier", "reject")] = tiers.get(sc.get("tier", "reject"), 0) + 1
        if sc.get("hard_fail"):
            hard_fails += 1

    print("\n" + "=" * 60)
    print("SNDK DETECTOR — RUN SUMMARY")
    print("=" * 60)
    print(f"Run timestamp : {state.get('run_timestamp', '')}")
    print(f"Candidates    : {len(candidates)} found")
    if by_source:
        for source, count in sorted(by_source.items()):
            print(f"                - {source}: {count}")
    print(f"Scored        : {len(scored)} (spinoffs evaluated this run)")
    print(
        f"Tiers         : starter={tiers['starter']} deep_dive={tiers['deep_dive']} "
        f"watchlist={tiers['watchlist']} reject={tiers['reject']}"
    )
    print(f"Hard fails    : {hard_fails}")
    print(f"Alerted       : {len(alerted)} (deep_dive/starter only)")

    if scored:
        print("\nTop scored:")
        ranked = sorted(
            scored,
            key=lambda c: (c.get("scorecard") or {}).get("total_score", 0),
            reverse=True,
        )
        for cand in ranked[:10]:
            sc = cand.get("scorecard") or {}
            flag = "🔔" if cand.get("alerted") else ("✗" if sc.get("hard_fail") else "  ")
            print(
                f"  {flag} {sc.get('total_score', 0):>3}/100  {sc.get('tier','?'):<10} "
                f"{cand.get('ticker'):<10} {cand.get('company_name', '')[:36]}"
            )

    print(f"\nErrors        : {len(errors)}")
    for err in errors:
        print(f"  ! {err}")
    print("=" * 60 + "\n")


def _run_and_report(config: Config, graph) -> AgentState:
    started = datetime.now(timezone.utc)
    logger.info("Run starting at %s", started.isoformat())
    state = asyncio.run(run_once(config, graph))
    _print_summary(state)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    logger.info("Run finished in %.1fs", elapsed)
    return state


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sndk-detector",
        description="Personal stock-screening agent (ingest -> score -> alert).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run a single pass and exit.")
    mode.add_argument(
        "--schedule",
        action="store_true",
        help="Run continuously, sleeping between passes.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=6.0,
        help="Hours between runs in --schedule mode (default: 6).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    # Fail loudly and early if config is incomplete.
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    init_db(config.db_path)
    logger.info("DB initialized at %s", config.db_path)
    graph = build_graph(config)

    if args.once:
        _run_and_report(config, graph)
        return 0

    # --schedule: simple sleep loop, no cron dependency.
    interval_seconds = max(1.0, args.interval * 3600)
    logger.info("Scheduling every %.1f hours. Ctrl-C to stop.", args.interval)
    try:
        while True:
            try:
                _run_and_report(config, graph)
            except Exception as exc:  # noqa: BLE001 - keep the scheduler alive
                logger.exception("Run failed, will retry next interval: %s", exc)
            logger.info("Sleeping %.1f hours until next run.", args.interval)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        return 0


if __name__ == "__main__":
    raise SystemExit(cli())
