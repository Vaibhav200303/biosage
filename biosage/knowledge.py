"""Loading and validating BioSage's local evidence corpus."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .models import EvidenceRecord


DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge.jsonl"


@lru_cache(maxsize=4)
def load_knowledge_base(path: str | Path = DEFAULT_KNOWLEDGE_PATH) -> tuple[EvidenceRecord, ...]:
    """Load and validate JSONL once per path.

    Failing fast on malformed records protects the grounded-answer pipeline: a bad
    source is caught at startup/test time rather than silently cited in a response.
    """

    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Evidence corpus not found: {corpus_path}")
    records: list[EvidenceRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            record = EvidenceRecord.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"Invalid evidence record at {corpus_path}:{line_number}: {exc}") from exc
        if record.evidence_id in seen:
            raise ValueError(f"Duplicate evidence_id {record.evidence_id} at line {line_number}")
        seen.add(record.evidence_id)
        records.append(record)
    if not records:
        raise ValueError(f"Evidence corpus is empty: {corpus_path}")
    return tuple(records)


def validate_corpus(records: Iterable[EvidenceRecord]) -> None:
    """Validate uniqueness and minimum traceability for an in-memory corpus."""

    records = tuple(records)
    ids = [record.evidence_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Evidence IDs must be unique")
    if any(not record.source_locator.strip() for record in records):
        raise ValueError("Every evidence record needs a source locator")
