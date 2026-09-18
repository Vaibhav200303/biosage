"""Small, inspectable TF-IDF retriever with metadata-aware filtering.

The optional scikit-learn implementation is used in deployment. A dependency-free
cosine fallback keeps the demo and unit tests usable before dependencies are installed,
while preserving the same trace shape and ranking semantics.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .knowledge import load_knowledge_base
from .models import EvidenceRecord, RetrievedEvidence

try:  # pragma: no cover - exercised when optional deployment dependency is installed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - covered by fallback tests indirectly
    TfidfVectorizer = None
    cosine_similarity = None


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")


def _tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


def _search_text(record: EvidenceRecord) -> str:
    return " ".join(
        [
            record.title,
            record.passage,
            " ".join(record.topics),
            " ".join(record.metrics),
            " ".join(record.conditions),
            " ".join(record.practices),
            " ".join(record.applicability),
        ]
    )


@dataclass(frozen=True)
class RetrievalTrace:
    lexical_score: float
    tag_score: float
    applicability_score: float

    @property
    def final_score(self) -> float:
        """Weighted relevance score; this is not a probability or confidence."""

        return min(1.0, 0.65 * self.lexical_score + 0.2 * self.tag_score + 0.15 * self.applicability_score)


class EvidenceRetriever:
    """Retrieve evidence by lexical similarity and optional metadata constraints."""

    def __init__(self, records: Iterable[EvidenceRecord] | None = None):
        self.records = tuple(load_knowledge_base() if records is None else records)
        if not self.records:
            raise ValueError("EvidenceRetriever requires at least one evidence record")
        self._documents = tuple(_search_text(record) for record in self.records)
        self._vectorizer = None
        self._matrix = None
        self._fallback_vectors: tuple[dict[str, float], ...] = ()
        self._fallback_idf: dict[str, float] = {}
        if TfidfVectorizer is not None:
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self._matrix = self._vectorizer.fit_transform(self._documents)
        else:
            self._fallback_vectors, self._fallback_idf = self._build_fallback_index(self._documents)

    @staticmethod
    def _build_fallback_index(documents: Iterable[str]) -> tuple[tuple[dict[str, float], ...], dict[str, float]]:
        tokenized = [_tokens(document) for document in documents]
        document_frequency = Counter(token for document in tokenized for token in set(document))
        count = max(1, len(tokenized))
        idf = {token: math.log((count + 1) / (frequency + 1)) + 1 for token, frequency in document_frequency.items()}
        vectors: list[dict[str, float]] = []
        for document in tokenized:
            counts = Counter(document)
            vector = {token: (1 + math.log(freq)) * idf[token] for token, freq in counts.items()}
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1
            vectors.append({token: value / norm for token, value in vector.items()})
        return tuple(vectors), idf

    def _fallback_transform(self, text: str) -> dict[str, float]:
        counts = Counter(_tokens(text))
        vector = {token: (1 + math.log(freq)) * self._fallback_idf[token] for token, freq in counts.items() if token in self._fallback_idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {token: value / norm for token, value in vector.items()}

    def _lexical_scores(self, query: str) -> list[float]:
        if self._vectorizer is not None and self._matrix is not None:
            query_vector = self._vectorizer.transform([query])
            return [max(0.0, float(value)) for value in cosine_similarity(query_vector, self._matrix)[0]]
        query_vector = self._fallback_transform(query)
        return [max(0.0, sum(query_vector.get(token, 0.0) * value for token, value in vector.items())) for vector in self._fallback_vectors]

    @staticmethod
    def _metadata_score(record: EvidenceRecord, query: str, metrics: set[str], practices: set[str], conditions: set[str]) -> RetrievalTrace:
        query_terms = set(_tokens(query))
        tags = set(record.metrics) | set(record.practices) | set(record.conditions) | set(record.topics)
        requested = metrics | practices | conditions
        tag_score = len(requested & tags) / len(requested) if requested else 0.0
        tag_tokens = {token for tag in tags for token in _tokens(tag)}
        query_tag_hits = len(query_terms & tag_tokens) / max(1, len(query_terms))
        applicability_tags = set(record.conditions) | set(record.applicability)
        applicability_score = len(conditions & applicability_tags) / len(conditions) if conditions else 0.0
        return RetrievalTrace(lexical_score=0.0, tag_score=max(tag_score, query_tag_hits), applicability_score=applicability_score)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
        metrics: Iterable[str] | None = None,
        conditions: Iterable[str] | None = None,
        practices: Iterable[str] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedEvidence]:
        """Return traceable records ranked by lexical + tag/applicability score.

        Metadata arguments are soft filters: matching records are ranked first, but a
        sparse query can still return useful adjacent evidence instead of an empty result.
        """

        if not query.strip():
            return []
        top_k = max(1, top_k)
        metric_set = {value.strip().lower() for value in (metrics or ()) if value.strip()}
        condition_set = {value.strip().lower() for value in (conditions or ()) if value.strip()}
        practice_set = {value.strip().lower() for value in (practices or ()) if value.strip()}
        lexical = self._lexical_scores(query)
        ranked: list[RetrievedEvidence] = []
        for index, record in enumerate(self.records):
            trace = self._metadata_score(record, query, metric_set, practice_set, condition_set)
            trace = RetrievalTrace(lexical[index], trace.tag_score, trace.applicability_score)
            requested = metric_set | condition_set | practice_set
            matched_tags = sorted(requested & (set(record.metrics) | set(record.conditions) | set(record.practices)))
            score = trace.final_score
            if requested and not matched_tags:
                score *= 0.65
            if score < min_score:
                continue
            query_terms = set(_tokens(query))
            matched_terms = sorted(query_terms & set(_tokens(_search_text(record))))
            ranked.append(
                RetrievedEvidence(
                    evidence=record,
                    score=round(score, 6),
                    lexical_score=round(trace.lexical_score, 6),
                    tag_score=round(trace.tag_score, 6),
                    applicability_score=round(trace.applicability_score, 6),
                    matched_terms=matched_terms,
                    filter_matches=matched_tags,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.evidence.evidence_id))
        return ranked[:top_k]


@lru_cache(maxsize=4)
def get_retriever(path: str | None = None) -> EvidenceRetriever:
    """Cache one retriever/index per corpus path for Streamlit reruns."""

    return EvidenceRetriever(load_knowledge_base(path) if path else load_knowledge_base())
