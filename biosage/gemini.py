"""Lazy Gemini synthesis with strict grounding and deterministic fallback."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from .config import get_settings
from .models import AssessmentResponse, EnvironmentalProfile, RetrievedEvidence
from .reasoning import assess
from .retrieval import EvidenceRetriever


class GeminiGenerationError(RuntimeError):
    """Raised when Gemini cannot produce a grounded, schema-valid response."""


class GeminiTimeoutError(GeminiGenerationError):
    """A locally enforced timeout; do not retry while the SDK worker may still run."""


UNSAFE_DRAFT_RE = re.compile(r"(?:\d|%|\bpercent(?:age)?\b|\bfold\b|\bdouble(?:s|d)?\b|\btriple(?:s|d)?\b|\btwice\b|\bhalf\b)", re.IGNORECASE)


class GeminiDraft(BaseModel):
    """Small synthesis contract; canonical recommendations/evidence stay local."""

    model_config = ConfigDict(extra="forbid")

    profile_summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(min_length=3)
    evidence_ids: list[str] = Field(min_length=1)


def build_grounded_prompt(profile: EnvironmentalProfile, evidence: Iterable[RetrievedEvidence], memory: list[dict[str, str]], seed: AssessmentResponse | None = None) -> str:
    evidence_lines = []
    for item in evidence:
        record = item.evidence
        evidence_lines.append(f"{record.evidence_id}: {record.title} | {record.passage} | source={record.source_locator} | url={record.url}")
    seed_text = seed.model_dump_json(exclude_none=True) if seed else "No seed supplied."
    safe_memory = [item for item in memory[-6:] if item.get("role") in {"user", "assistant"}]
    return """You are BioSage, an evidence-grounded land restoration assistant.
Use only the supplied profile, deterministic seed, and evidence records. Do not add or alter recommendations, confidence values, or evidence records.
Return only JSON matching the small GeminiDraft schema: profile_summary, assumptions, reasoning_chain (at least 3 strings), and evidence_ids.
Copy every evidence ID cited by the deterministic seed into evidence_ids, and copy IDs exactly from the supplied evidence IDs. Never invent citations or numeric effects.

PROFILE:
{profile}

SUPPLIED EVIDENCE (the only allowed citation IDs):
{evidence}

RECENT MEMORY:
{memory}

DETERMINISTIC SEED TO PRESERVE:
{seed}
""".format(profile=profile.model_dump_json(exclude_none=True), evidence="\n".join(evidence_lines), memory=json.dumps(safe_memory), seed=seed_text)


def validate_grounding(response: AssessmentResponse, supplied_ids: set[str]) -> AssessmentResponse:
    """Fail closed if Gemini returns unknown or duplicate evidence identifiers."""

    response_ids = [item.evidence_id for item in response.evidence]
    if len(response_ids) != len(set(response_ids)):
        raise GeminiGenerationError("Gemini returned duplicate evidence IDs")
    if not set(response_ids) <= supplied_ids:
        raise GeminiGenerationError("Gemini returned evidence outside the supplied subset")
    cited = {evidence_id for recommendation in response.recommendations for evidence_id in recommendation.evidence_ids}
    if not cited <= supplied_ids:
        raise GeminiGenerationError("Gemini cited evidence outside the supplied subset")
    return response


def validate_draft_grounding(draft: GeminiDraft, supplied_ids: set[str]) -> GeminiDraft:
    if not set(draft.evidence_ids) <= supplied_ids:
        raise GeminiGenerationError("Gemini draft cited evidence outside the supplied subset")
    draft_text = " ".join([draft.profile_summary, *draft.assumptions, *draft.reasoning_chain])
    if UNSAFE_DRAFT_RE.search(draft_text):
        raise GeminiGenerationError("Gemini draft contains an unsupported numeric or effect claim")
    return draft


class GeminiSynthesizer:
    """Injectable wrapper; importing google-genai is deferred until first use."""

    def __init__(self, client: Any | None = None, model: str | None = None, max_retries: int = 2, timeout_seconds: float = 20):
        self.client = client
        self.model = model or get_settings().gemini_model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        settings = get_settings()
        if not settings.gemini_api_key:
            raise GeminiGenerationError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
            self.client = genai.Client(api_key=settings.gemini_api_key)
        except Exception as exc:
            raise GeminiGenerationError("Gemini SDK is unavailable") from exc
        return self.client

    def _call(self, client: Any, prompt: str) -> Any:
        """Bound a potentially blocking SDK call without importing the SDK at module load."""

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            client.models.generate_content,
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_json_schema": GeminiDraft.model_json_schema()},
        )
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise GeminiTimeoutError(f"Gemini call exceeded {self.timeout_seconds}s timeout") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def generate(self, profile: EnvironmentalProfile, evidence: list[RetrievedEvidence], memory: list[dict[str, str]] | None = None, seed: AssessmentResponse | None = None) -> AssessmentResponse:
        if seed is None or seed.status != "complete":
            raise GeminiGenerationError("Gemini synthesis requires a complete deterministic seed")
        allowed_ids = {item.evidence_id for item in evidence}
        prompt = build_grounded_prompt(profile, evidence, memory or [], seed)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                client = self._client()
                response = self._call(client, prompt)
                raw = getattr(response, "text", response)
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                draft = validate_draft_grounding(GeminiDraft.model_validate(parsed), allowed_ids)
                seed_cited = {evidence_id for recommendation in seed.recommendations for evidence_id in recommendation.evidence_ids}
                if not seed_cited <= set(draft.evidence_ids):
                    raise GeminiGenerationError("Gemini draft omitted a seed citation")
                merged = seed.model_dump()
                merged["profile_summary"] = draft.profile_summary or seed.profile_summary
                merged["assumptions"] = list(seed.assumptions)
                merged["reasoning_chain"] = draft.reasoning_chain
                return AssessmentResponse.model_validate(merged)
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                retryable = not isinstance(exc, GeminiTimeoutError) and any(token in message for token in ("429", "quota", "rate", "timeout", "temporarily"))
                if attempt >= self.max_retries or not retryable:
                    break
                time.sleep(0.25 * (2**attempt))
        raise GeminiGenerationError(f"Gemini response rejected: {last_error}") from last_error


def generate_assessment(profile: EnvironmentalProfile, *, query: str = "", retriever: EvidenceRetriever | None = None, memory: list[dict[str, str]] | None = None, synthesizer: GeminiSynthesizer | None = None) -> AssessmentResponse:
    """Try Gemini only after local retrieval; always retain a deterministic fallback."""

    retriever = retriever or EvidenceRetriever()
    seed = assess(profile, query=query, retriever=retriever)
    if seed.status != "complete":
        return seed
    evidence = seed.evidence
    if synthesizer is not None or get_settings().gemini_api_key:
        try:
            return (synthesizer or GeminiSynthesizer()).generate(profile, evidence, memory, seed)
        except GeminiGenerationError:
            pass
    return seed
