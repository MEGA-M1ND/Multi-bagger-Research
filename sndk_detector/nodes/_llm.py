"""Shared LLM plumbing for the v2 evidence nodes.

Lifted out of the old score_blueprint.py so every extraction/synthesis node
reuses the same client factory, prompt loader, retry/backoff, and — critically —
the snippet anti-hallucination guard that enforces "the LLM may only quote text
that actually exists in the source document".
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional, Type, TypeVar

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from ..config import Config
from ..schemas import EvidenceField
from ..state import Candidate

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_MAX_RETRIES = 4
_BASE_BACKOFF = 1.5  # seconds; grows ~ _BASE_BACKOFF ** attempt

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load and cache a prompt template from prompts/."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def get_client(config: Config) -> AsyncOpenAI:
    """A fresh AsyncOpenAI client for a run."""
    return AsyncOpenAI(api_key=config.openai_api_key)


async def call_with_retry(coro_factory: Callable, what: str):
    """Await an OpenAI call, retrying transient failures with backoff.

    ``coro_factory`` is a zero-arg callable returning a fresh awaitable each
    attempt (a spent coroutine can't be re-awaited).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await coro_factory()
        except (RateLimitError, APIConnectionError) as exc:
            last_exc = exc
            delay = _BASE_BACKOFF ** (attempt + 1) + random.uniform(0, 0.5)
            logger.warning(
                "%s: transient error (attempt %d/%d), backing off %.1fs: %s",
                what, attempt + 1, _MAX_RETRIES, delay, exc,
            )
            await asyncio.sleep(delay)
        except APIError as exc:
            last_exc = exc
            if attempt >= _MAX_RETRIES - 1:
                break
            delay = _BASE_BACKOFF ** (attempt + 1) + random.uniform(0, 0.5)
            logger.warning("%s: API error, retrying in %.1fs: %s", what, delay, exc)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def extract_model(
    client: AsyncOpenAI,
    config: Config,
    prompt: str,
    model_cls: Type[T],
    what: str,
    temperature: float = 0.1,
) -> T:
    """Call the LLM expecting JSON, validate it into ``model_cls``.

    Raises on terminal failure (caller catches and records as an error string,
    mirroring _score_one's discipline).
    """
    resp = await call_with_retry(
        lambda: client.chat.completions.create(
            model=config.openai_model,
            response_format={"type": "json_object"},
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ),
        what=what,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return model_cls.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(f"{what}: schema validation failed: {exc}") from exc


def source_texts_of(candidate: Candidate) -> Dict[str, str]:
    """Map {source_id -> fetched_text} for this candidate's source documents.

    Used by guard_snippets to verify the LLM only quoted real filing text.
    """
    raw = candidate.get("raw_data") or {}
    out: Dict[str, str] = {}
    for doc in raw.get("source_documents") or []:
        sid = doc.get("source_id")
        if sid and doc.get("fetched_text"):
            out[sid] = doc["fetched_text"]
    return out


def render_candidate_block(candidate: Candidate) -> str:
    """Render everything known about a candidate into an LLM-friendly block.

    Includes identity, source documents (with their source_id so the LLM can
    cite them), market data, news, and any evidence extracted by earlier nodes.
    Only non-empty sections are included to keep the prompt tight.
    """
    raw = candidate.get("raw_data") or {}
    block: dict = {
        "ticker": candidate.get("ticker"),
        "company_name": candidate.get("company_name"),
        "cik": candidate.get("cik"),
        "market": candidate.get("market"),
    }
    if candidate.get("event_family"):
        block["event_family"] = candidate["event_family"]

    docs = []
    for doc in raw.get("source_documents") or []:
        docs.append({
            "source_id": doc.get("source_id"),
            "form": doc.get("form"),
            "file_date": doc.get("file_date"),
            "text": doc.get("fetched_text"),
        })
    if docs:
        block["source_documents"] = docs

    # Market data + news pulled by ingestion / yfinance enrichment.
    for key in ("yf_data", "yf_news", "headline", "summary", "filing_text"):
        if raw.get(key):
            block[key] = raw[key]

    # Evidence from earlier nodes (so extract/synthesize/critic see prior context).
    for key in ("event_signal", "financial_snapshot", "moat_proxy",
                "valuation_gap", "risk_flags", "scorecard", "memo"):
        if candidate.get(key):
            block[key] = candidate[key]

    return json.dumps(block, indent=2, default=str)


def evidence_rows(
    candidate_id: str, kind: str, model: BaseModel, provenance: str = "llm_filing"
) -> list:
    """Flatten a pydantic evidence model into db.upsert_evidence row dicts.

    One row per top-level EvidenceField, keyed by a stable evidence_id so
    re-extraction is idempotent.
    """
    import hashlib
    rows = []
    for field_name in model.__class__.model_fields:
        value = getattr(model, field_name, None)
        if not isinstance(value, EvidenceField):
            continue
        eid = hashlib.sha256(
            f"{candidate_id}:{kind}:{field_name}".encode("utf-8")
        ).hexdigest()[:16]
        rows.append({
            "evidence_id": eid,
            "kind": kind,
            "field": field_name,
            "value_json": json.dumps(value.value, default=str),
            "confidence": value.confidence,
            "source_id": value.source_id,
            "provenance": provenance,
            "snippet": value.snippet,
            "extractor_version": value.extractor_version,
        })
    return rows


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for tolerant substring matching."""
    return _WS_RE.sub(" ", (text or "").lower()).strip()


def guard_snippets(model: BaseModel, source_texts: Dict[str, str]) -> int:
    """Zero out the confidence of any EvidenceField whose snippet is NOT a
    verbatim substring of its cited source document.

    This is the cheap, deterministic anti-hallucination guard: the LLM is told
    to quote exactly, and anything it can't back with real source text is
    demoted to confidence 0 (excluded from scoring). Returns the count of
    fields demoted. Operates in place on top-level EvidenceField attributes.
    """
    demoted = 0
    norm_sources = {sid: _normalize(txt) for sid, txt in source_texts.items() if txt}
    for field_name in model.__class__.model_fields:
        value = getattr(model, field_name, None)
        if not isinstance(value, EvidenceField):
            continue
        snippet = value.snippet
        if not snippet or not snippet.strip():
            continue  # no claim to a quote -> nothing to verify
        sid = value.source_id
        haystack = norm_sources.get(sid) if sid else None
        # If we have the cited source and the snippet isn't in it -> demote.
        # If the snippet cites no/unknown source, we also can't trust it.
        if not haystack or _normalize(snippet) not in haystack:
            value.confidence = 0.0
            demoted += 1
    return demoted
