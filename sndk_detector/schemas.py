"""Pydantic models for evidence contracts.

These validate the JSON produced by the LLM extraction/synthesis prompts. The
whole point of v2 is to demote the LLM from "judge of truth" to "extractor /
writer over structured evidence" — so every claim it makes must fit one of
these shapes, carry provenance, and default to "unknown" (None) when the
evidence is absent.

We use these ONLY to validate LLM output (parse + retry on ValidationError,
mirroring the spirit of ``_coerce_blueprint`` in the old scorer). The LangGraph
state itself stays TypedDict (see state.py) to match the existing AgentState /
Candidate conventions.

Confidence rubric (applies to every EvidenceField.confidence):
    0.90 - 1.00  explicit filing language
    0.70 - 0.89  inferred from multiple data points
    0.50 - 0.69  weak — downweighted in scoring
    < 0.50       memo-only note, excluded from scoring
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class EvidenceField(BaseModel):
    """One atomic, auditable claim.

    ``value is None`` means "unknown / not stated" — this is the discipline that
    keeps the LLM from inventing facts. ``snippet`` must be a verbatim substring
    of the source document when ``source_id`` is set (the caller verifies this
    and zeroes out hallucinated snippets).
    """

    value: Optional[Any] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_id: Optional[str] = None
    snippet: Optional[str] = None
    extractor_version: str = "v1"


EventFamily = Literal["spinoff", "carveout", "unknown"]


class EventSignal(BaseModel):
    """Classification + key facts of the structural event."""

    event_family: EventFamily = "unknown"
    parent_entity: EvidenceField = Field(default_factory=EvidenceField)
    spun_entity: EvidenceField = Field(default_factory=EvidenceField)
    record_date: EvidenceField = Field(default_factory=EvidenceField)
    distribution_ratio: EvidenceField = Field(default_factory=EvidenceField)
    rationale: EvidenceField = Field(default_factory=EvidenceField)


class FinancialSnapshot(BaseModel):
    """Objective financial state. Either yfinance-sourced or LLM-extracted."""

    revenue_ttm: EvidenceField = Field(default_factory=EvidenceField)
    ebitda_ttm: EvidenceField = Field(default_factory=EvidenceField)
    gross_margin_ttm: EvidenceField = Field(default_factory=EvidenceField)
    cash: EvidenceField = Field(default_factory=EvidenceField)
    total_debt: EvidenceField = Field(default_factory=EvidenceField)
    net_debt: EvidenceField = Field(default_factory=EvidenceField)
    nearest_debt_maturity: EvidenceField = Field(default_factory=EvidenceField)
    fcf_ttm: EvidenceField = Field(default_factory=EvidenceField)
    shares_out: EvidenceField = Field(default_factory=EvidenceField)


class MoatProxy(BaseModel):
    """Hard moat proxies — competitors can't replicate in < 2 years."""

    switching_costs: EvidenceField = Field(default_factory=EvidenceField)
    customer_concentration: EvidenceField = Field(default_factory=EvidenceField)
    contractual_durability: EvidenceField = Field(default_factory=EvidenceField)
    market_position: EvidenceField = Field(default_factory=EvidenceField)
    capital_intensity: EvidenceField = Field(default_factory=EvidenceField)


class RiskFlag(BaseModel):
    """Disqualifier signals. Each value is a bool (True == flag is present)."""

    refinancing_12mo: EvidenceField = Field(default_factory=EvidenceField)
    material_dilution: EvidenceField = Field(default_factory=EvidenceField)
    customer_concentration_risk: EvidenceField = Field(default_factory=EvidenceField)
    governance_red_flag: EvidenceField = Field(default_factory=EvidenceField)
    no_catalyst_path: EvidenceField = Field(default_factory=EvidenceField)


class ValuationGap(BaseModel):
    """Peer-relative valuation. Computed deterministically (not by the LLM)."""

    subject_multiple: EvidenceField = Field(default_factory=EvidenceField)
    peer_median_multiple: EvidenceField = Field(default_factory=EvidenceField)
    gap_pct: EvidenceField = Field(default_factory=EvidenceField)
    peer_set: EvidenceField = Field(default_factory=EvidenceField)
    multiple_kind: str = "EV/EBITDA"


class Scorecard(BaseModel):
    """The 100-point weighted score. Produced deterministically in scoring.py."""

    event_quality: int = Field(default=0, ge=0, le=20)
    cycle_position: int = Field(default=0, ge=0, le=15)
    secular_tailwind: int = Field(default=0, ge=0, le=20)
    moat_proxies: int = Field(default=0, ge=0, le=15)
    valuation_dislocation: int = Field(default=0, ge=0, le=20)
    survivability: int = Field(default=0, ge=0, le=15)
    total_score: int = Field(default=0, ge=0, le=100)
    hard_fail: bool = False
    hard_fail_reasons: List[str] = Field(default_factory=list)
    tier: Literal["reject", "watchlist", "deep_dive", "starter"] = "reject"


class Memo(BaseModel):
    """The investment memo. LLM-written, but only over the evidence bundle."""

    why_now: str = "unknown"
    why_mispriced: str = "unknown"
    what_must_happen_12m: str = "unknown"
    bear_case: str = "unknown"
    disconfirming_evidence: List[str] = Field(default_factory=list)
    next_checkpoints: List[str] = Field(default_factory=list)


class Critic(BaseModel):
    """Adversarial second pass that tries to break the thesis."""

    unsupported_claims: List[str] = Field(default_factory=list)
    double_counted_factors: List[str] = Field(default_factory=list)
    short_seller_view: str = "unknown"
    most_likely_invalidating_metric: str = "unknown"
