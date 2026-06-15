"""Alert node: notify on the highest-conviction names only.

v2 filter rule: alert a candidate iff
  * its tier is 'deep_dive' or 'starter' (the only tiers worth interrupting for), AND
  * we have not already alerted on it (checked against the DB by candidate_id).

This is a deliberate shift from v1's "any score >= threshold" — most scored names
land in reject/watchlist and live in the files, not your phone.

Delivery is Telegram (Markdown). Send failures are caught and recorded; we only
``mark_as_alerted`` after a confirmed successful send, so a failed send is retried
on the next run (idempotency without double-alerting).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import requests

from ..config import Config
from ..db import has_been_alerted, mark_as_alerted
from ..state import Candidate

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_REQUEST_TIMEOUT = 20

# Telegram messages cap at 4096 chars; leave headroom for Markdown overhead.
_MAX_MESSAGE_LEN = 3800

# Tiers that earn a push notification.
_ALERT_TIERS = ("deep_dive", "starter")


def _escape_md(text: str) -> str:
    """Escape the few characters that break legacy Markdown parse mode."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, "\\" + ch)
    return text


def format_alert(candidate: Candidate) -> str:
    """Render a candidate into a Telegram Markdown message (v2: score/100 + tier)."""
    sc = candidate.get("scorecard") or {}
    memo = candidate.get("memo") or {}
    critic = candidate.get("critic") or {}

    ticker = _escape_md(str(candidate.get("ticker", "?")))
    name = _escape_md(str(candidate.get("company_name", "")))
    total = sc.get("total_score", 0)
    tier = _escape_md(str(sc.get("tier", "?")))

    lines = [
        f"*SNDK {tier.upper()}: {ticker}* — {name}",
        f"Score: *{total}/100* · {_escape_md(str(candidate.get('event_family', '?')))} · "
        f"{_escape_md(str(candidate.get('market', '?')))}",
        "",
        "*Subscores:*",
        f"event {sc.get('event_quality',0)} · cycle {sc.get('cycle_position',0)} · "
        f"tailwind {sc.get('secular_tailwind',0)} · moat {sc.get('moat_proxies',0)} · "
        f"valuation {sc.get('valuation_dislocation',0)} · survivability {sc.get('survivability',0)}",
    ]

    if memo.get("why_now"):
        lines += ["", "*Why now:*", _escape_md(str(memo["why_now"]).strip())]
    if memo.get("why_mispriced"):
        lines += ["", "*Why mispriced:*", _escape_md(str(memo["why_mispriced"]).strip())]
    if critic.get("most_likely_invalidating_metric"):
        lines += ["", "*Watch (kill metric):*",
                  _escape_md(str(critic["most_likely_invalidating_metric"]).strip())]

    message = "\n".join(lines)
    if len(message) > _MAX_MESSAGE_LEN:
        message = message[:_MAX_MESSAGE_LEN] + "…"
    return message


def _send_telegram(config: Config, message: str) -> Tuple[bool, str]:
    """Send one Telegram message. Returns (ok, detail). Never raises."""
    url = _TELEGRAM_API.format(token=config.telegram_bot_token)
    payload = {
        "chat_id": config.telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return False, f"network error: {exc}"

    if resp.status_code == 200 and resp.json().get("ok"):
        return True, "ok"
    return False, f"telegram returned {resp.status_code}: {resp.text[:200]}"


def select_alertable(config: Config, decided: List[Candidate]) -> List[Candidate]:
    """Filter to deep_dive/starter tier candidates not yet alerted."""
    queue: List[Candidate] = []
    for cand in decided:
        tier = (cand.get("scorecard") or {}).get("tier")
        if tier not in _ALERT_TIERS:
            continue
        if has_been_alerted(config.db_path, cand["candidate_id"]):
            continue
        queue.append(cand)
    return queue


def run_alerts(
    config: Config, decided: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    """Send alerts for qualifying candidates. Returns (alerted, errors)."""
    queue = select_alertable(config, decided)
    alerted: List[Candidate] = []
    errors: List[str] = []

    for cand in queue:
        message = format_alert(cand)
        ok, detail = _send_telegram(config, message)
        if ok:
            cand["alerted"] = True
            try:
                mark_as_alerted(config.db_path, cand, message)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"alert: db mark failed for {cand.get('ticker')}: {exc}")
            alerted.append(cand)
            logger.info("alert: sent for %s", cand.get("ticker"))
        else:
            # Not marked as alerted -> will be retried next run.
            errors.append(f"alert: send failed for {cand.get('ticker')}: {detail}")
            logger.warning("alert: send failed for %s: %s", cand.get("ticker"), detail)

    return alerted, errors


def make_alert_node(config: Config):
    """Factory: returns a graph node that closes over ``config``."""

    def _node(state: dict) -> dict:
        alerted, errors = run_alerts(config, state.get("decided_candidates", []))
        return {"alert_queue": alerted, "errors": errors}

    return _node
