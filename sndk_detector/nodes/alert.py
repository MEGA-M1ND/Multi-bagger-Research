"""Alert node: filter to high-scoring, not-yet-alerted candidates and notify.

Filter rule: alert a candidate iff
  * it has a blueprint AND total_score >= BLUEPRINT_THRESHOLD, AND
  * we have not already alerted on it (checked against the DB by candidate_id).

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

_FACTOR_LABELS = {
    "structural_event": "Structural event",
    "cyclical_trough": "Cyclical trough",
    "secular_tailwind": "Secular tailwind",
    "supply_constraint": "Supply constraint",
    "undervalued_narrative": "Undervalued narrative",
    "domain_edge": "Domain edge",
}


def _escape_md(text: str) -> str:
    """Escape the few characters that break legacy Markdown parse mode."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, "\\" + ch)
    return text


def format_alert(candidate: Candidate, threshold: int) -> str:
    """Render a candidate into a Telegram Markdown message."""
    bp = candidate.get("blueprint") or {}
    ticker = _escape_md(str(candidate.get("ticker", "?")))
    name = _escape_md(str(candidate.get("company_name", "")))
    score = bp.get("total_score", 0)

    lines = [
        f"*SNDK Match: {ticker}* — {name}",
        f"Score: *{score}/6* (threshold {threshold}) · "
        f"{_escape_md(str(candidate.get('market', '?')))} · "
        f"{_escape_md(str(candidate.get('source', '?')))}",
        "",
        "*Blueprint:*",
    ]
    for key, label in _FACTOR_LABELS.items():
        mark = "✅" if bp.get(key) else "▫️"
        lines.append(f"{mark} {label}")

    thesis = candidate.get("thesis")
    if thesis:
        lines += ["", "*Thesis:*", _escape_md(thesis.strip())]
    elif bp.get("reasoning"):
        lines += ["", "*Reasoning:*", _escape_md(str(bp["reasoning"]).strip())]

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


def select_alertable(config: Config, scored: List[Candidate]) -> List[Candidate]:
    """Filter scored candidates down to those we should alert on."""
    queue: List[Candidate] = []
    for cand in scored:
        bp = cand.get("blueprint")
        if not bp or bp.get("total_score", 0) < config.blueprint_threshold:
            continue
        if has_been_alerted(config.db_path, cand["candidate_id"]):
            continue
        queue.append(cand)
    return queue


def run_alerts(
    config: Config, scored: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    """Send alerts for qualifying candidates. Returns (alerted, errors)."""
    queue = select_alertable(config, scored)
    alerted: List[Candidate] = []
    errors: List[str] = []

    for cand in queue:
        message = format_alert(cand, config.blueprint_threshold)
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
        alerted, errors = run_alerts(config, state.get("scored_candidates", []))
        return {"alert_queue": alerted, "errors": errors}

    return _node
