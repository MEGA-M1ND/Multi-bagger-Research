"""Configuration loading + validation.

All runtime config comes from environment variables (loaded from a local
``.env`` via python-dotenv). Required keys are validated eagerly so the agent
fails loudly at startup rather than halfway through a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once on import. override=False so real environment variables
# (e.g. in CI or a systemd unit) take precedence over the file.
load_dotenv(override=False)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


# Keys that must be present for a full run (ingest -> score -> alert).
REQUIRED_KEYS = ("OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    openai_model: str
    telegram_bot_token: str
    telegram_chat_id: str
    blueprint_threshold: int
    dedup_lookback_days: int
    max_candidates_per_source: int
    max_concurrent_llm: int
    db_path: str
    sec_user_agent: str


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def load_config() -> Config:
    """Load and validate config. Raises ConfigError on missing required keys."""
    missing = [k for k in REQUIRED_KEYS if not (os.getenv(k) or "").strip()]
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ".\nCopy .env.example to .env and fill them in."
        )

    threshold = _get_int("BLUEPRINT_THRESHOLD", 4)
    if not 0 <= threshold <= 6:
        raise ConfigError(f"BLUEPRINT_THRESHOLD must be between 0 and 6, got {threshold}")

    max_concurrent = _get_int("MAX_CONCURRENT_LLM", 5)
    if max_concurrent < 1:
        raise ConfigError(f"MAX_CONCURRENT_LLM must be >= 1, got {max_concurrent}")

    return Config(
        openai_api_key=os.environ["OPENAI_API_KEY"].strip(),
        openai_model=(os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
        telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"].strip(),
        telegram_chat_id=os.environ["TELEGRAM_CHAT_ID"].strip(),
        blueprint_threshold=threshold,
        dedup_lookback_days=_get_int("DEDUP_LOOKBACK_DAYS", 7),
        max_candidates_per_source=_get_int("MAX_CANDIDATES_PER_SOURCE", 20),
        max_concurrent_llm=max_concurrent,
        db_path=(os.getenv("DB_PATH") or "sndk_detector.db").strip(),
        sec_user_agent=(
            os.getenv("SEC_USER_AGENT") or "SNDK Detector contact@example.com"
        ).strip(),
    )
