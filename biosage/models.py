"""Typed domain contracts shared by the UI, reasoning engine, and retriever."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class EnvironmentalProfile(BaseModel):
    """A farmer- or land-manager-provided description of a site.

    Values are optional because the first conversational turn may be incomplete. The
    reasoning layer can use ``provided_categories`` to decide whether clarification is
    required before generating an assessment.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    region: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    soil_ph: float | None = Field(default=None, ge=0, le=14)
    soil_organic_carbon_pct: float | None = Field(default=None, ge=0, le=100)
    soil_moisture: Literal["very_low", "low", "moderate", "high", "waterlogged"] | None = None
    soil_texture: Literal["sandy", "loamy", "clayey", "peaty", "unknown"] | None = None
    rainfall_mm_year: float | None = Field(default=None, ge=0)
    temperature_c_mean: float | None = None
    water_availability: Literal["scarce", "seasonal", "adequate", "excess"] | None = None
    land_use: str | None = None
    crop_system: str | None = None
    species_richness: float | None = Field(default=None, ge=0)
    habitat_diversity: str | None = None
    pollinator_presence: Literal["low", "moderate", "high", "unknown"] | None = None
    pollution_pressure: Literal["none", "low", "moderate", "high", "unknown"] | None = None
    deforestation_pressure: Literal["none", "low", "moderate", "high", "unknown"] | None = None
    fragmentation_pressure: Literal["none", "low", "moderate", "high", "unknown"] | None = None
    notes: str | None = None

    @property
    def provided_categories(self) -> set[str]:
        def meaningful(value: object) -> bool:
            return value is not None and str(value).lower() != "unknown"

        categories: set[str] = set()
        if meaningful(self.land_use) or meaningful(self.crop_system):
            categories.add("land_use")
        if any(
            meaningful(value)
            for value in (self.soil_ph, self.soil_organic_carbon_pct, self.soil_moisture, self.soil_texture)
        ):
            categories.add("soil")
        if any(meaningful(value) for value in (self.rainfall_mm_year, self.temperature_c_mean, self.water_availability)):
            categories.add("climate_water")
        if any(
            meaningful(value)
            for value in (
                self.species_richness,
                self.habitat_diversity,
                self.pollinator_presence,
                self.fragmentation_pressure,
            )
        ):
            categories.add("biodiversity")
        if any(meaningful(value) for value in (self.pollution_pressure, self.deforestation_pressure)):
            categories.add("human_pressure")
        return categories

    @property
    def is_sufficient_for_assessment(self) -> bool:
        """Require at least three useful categories before a full assessment."""

        return len(self.provided_categories) >= 3


class NumericClaim(BaseModel):
    """A numeric range that a source explicitly supports, including its qualifier."""

    model_config = ConfigDict(extra="forbid")

    lower: float | None = None
    upper: float | None = None
    unit: str
    time_horizon: str
    context: str
    source_locator: str
    metric: str
    direction: Literal["increase", "decrease", "range", "threshold"] = "range"
    caveat: str | None = None

    @model_validator(mode="after")
    def at_least_one_bound(self) -> "NumericClaim":
        if self.lower is None and self.upper is None:
            raise ValueError("a numeric claim needs at least a lower or upper bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("numeric claim lower bound cannot exceed upper bound")
        return self


class EvidenceRecord(BaseModel):
    """One short, paraphrased, traceable evidence item in the local corpus."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    evidence_id: str = Field(pattern=r"^E\d{3}$")
    title: str
    organization: str
    year: int | None = Field(default=None, ge=1900, le=2100)
    url: HttpUrl
    passage: str = Field(min_length=30)
    source_locator: str = Field(min_length=3)
    topics: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(min_length=1)
    conditions: list[str] = Field(default_factory=list)
    practices: list[str] = Field(default_factory=list)
    applicability: list[str] = Field(default_factory=list)
    numeric_claims: list[NumericClaim] = Field(default_factory=list)

    @field_validator("metrics", "conditions", "practices", "topics", "applicability")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        return sorted({value.strip().lower() for value in values if value.strip()})


class RetrievedEvidence(BaseModel):
    """Evidence plus the retrieval trace shown to users and used for grounding."""

    model_config = ConfigDict(extra="forbid")

    evidence: EvidenceRecord
    score: float = Field(ge=0, le=1)
    lexical_score: float = Field(ge=0, le=1)
    tag_score: float = Field(ge=0, le=1)
    applicability_score: float = Field(ge=0, le=1)
    matched_terms: list[str] = Field(default_factory=list)
    filter_matches: list[str] = Field(default_factory=list)

    @property
    def evidence_id(self) -> str:
        return self.evidence.evidence_id


class Recommendation(BaseModel):
    """Contract reserved for the reasoning milestone and future Gemini output."""

    model_config = ConfigDict(extra="forbid")

    action: str
    mechanism: str
    affected_metrics: list[str] = Field(min_length=1)
    implementation_steps: list[str] = Field(min_length=1)
    time_horizon: str
    confidence: Literal["low", "medium", "high"]
    evidence_ids: list[str] = Field(min_length=1)
    caveats: list[str] = Field(default_factory=list)
    measurable_indicators: list[str] = Field(default_factory=list)


class AssessmentResponse(BaseModel):
    """Top-level response contract for deterministic and Gemini-backed assessments."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["needs_clarification", "complete"]
    clarifying_questions: list[str] = Field(default_factory=list, max_length=3)
    profile_summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(default_factory=list)
    reasoning_variables: list[str] = Field(default_factory=list)
    evidence: list[RetrievedEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_contract(self) -> "AssessmentResponse":
        if self.status == "needs_clarification":
            if self.recommendations:
                raise ValueError("clarification responses cannot contain recommendations")
            if not self.clarifying_questions:
                raise ValueError("clarification responses need at least one question")
        else:
            if self.clarifying_questions:
                raise ValueError("complete responses cannot contain clarifying questions")
            if len(self.recommendations) < 3:
                raise ValueError("complete responses need at least three recommendations")
            if len(self.reasoning_chain) < 3:
                raise ValueError("complete responses need a multi-step reasoning chain")
            if len(set(self.reasoning_variables)) < 3:
                raise ValueError("complete responses need three distinct reasoning variables")
        return self
