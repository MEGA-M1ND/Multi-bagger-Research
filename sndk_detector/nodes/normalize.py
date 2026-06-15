"""Normalization node: entity resolution + provenance persistence.

Sits between the parallel ingestion fan-out and the evidence layers. It:
  1. Collapses duplicate candidates (same candidate_id from different ingestion
     branches, or an amended filing alongside its original), merging their
     source_documents so no provenance is lost.
  2. Resolves identity — ensures ``cik`` is set where we have it.
  3. Persists every source_document to the DB and records the linked
     ``source_doc_ids`` on the candidate so downstream nodes (and the memo's
     click-through links) can find the exact filing.

Defensive throughout: DB failures append to ``errors`` and never crash the graph,
mirroring the discipline in the ingestion nodes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from ..config import Config
from ..db import upsert_candidate, upsert_source_document
from ..state import Candidate

logger = logging.getLogger(__name__)


def _merge(into: Candidate, other: Candidate) -> None:
    """Merge ``other`` into ``into`` in place — union their source_documents."""
    into_raw = into.get("raw_data") or {}
    other_raw = other.get("raw_data") or {}
    docs = list(into_raw.get("source_documents") or [])
    seen = {d.get("source_id") for d in docs}
    for d in other_raw.get("source_documents") or []:
        if d.get("source_id") not in seen:
            docs.append(d)
            seen.add(d.get("source_id"))
    into_raw["source_documents"] = docs
    into["raw_data"] = into_raw
    # Prefer a real CIK if the merged candidate is missing one.
    if not into.get("cik") and other.get("cik"):
        into["cik"] = other["cik"]


def normalize_candidates(
    config: Config, candidates: List[Candidate]
) -> Tuple[List[Candidate], List[str]]:
    """Dedupe + persist source docs. Returns (normalized, errors)."""
    errors: List[str] = []
    by_id: Dict[str, Candidate] = {}

    # 1. Collapse duplicates by candidate_id, merging source_documents.
    for cand in candidates:
        cid = cand.get("candidate_id")
        if not cid:
            continue
        if cid in by_id:
            _merge(by_id[cid], cand)
        else:
            by_id[cid] = cand

    normalized: List[Candidate] = []
    for cand in by_id.values():
        raw = cand.get("raw_data") or {}
        docs = raw.get("source_documents") or []

        # Persist the candidate shell FIRST so the source_documents FK is
        # satisfied (foreign_keys=ON), and so the candidate is observable even
        # if it's later dropped at classification.
        try:
            upsert_candidate(config.db_path, cand)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"normalize: candidate persist failed for {cand.get('ticker')}: {exc}"
            )

        # Then persist source documents + record their ids on the candidate.
        doc_ids: List[str] = []
        for doc in docs:
            sid = doc.get("source_id")
            if not sid:
                continue
            try:
                upsert_source_document(config.db_path, cand["candidate_id"], doc)
                doc_ids.append(sid)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"normalize: source_document persist failed for "
                    f"{cand.get('ticker')}: {exc}"
                )
        cand["source_doc_ids"] = doc_ids
        normalized.append(cand)

    logger.info(
        "normalize: %d in -> %d unique candidates", len(candidates), len(normalized)
    )
    return normalized, errors


def make_normalize_node(config: Config):
    """Factory: returns a graph node that closes over ``config``."""

    def _node(state: dict) -> dict:
        normalized, errors = normalize_candidates(config, state.get("candidates", []))
        return {"normalized_candidates": normalized, "errors": errors}

    return _node
