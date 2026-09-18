from biosage.retrieval import EvidenceRetriever


def test_retrieval_returns_traceable_ranked_results():
    retriever = EvidenceRetriever()
    results = retriever.retrieve(
        "low soil organic carbon and water stress",
        top_k=5,
        metrics=["soil organic carbon", "soil moisture"],
        conditions=["water stress"],
    )
    assert 1 <= len(results) <= 5
    assert results == sorted(results, key=lambda result: (-result.score, result.evidence_id))
    assert all(0 <= result.score <= 1 for result in results)
    assert all(result.evidence_id.startswith("E") for result in results)


def test_metadata_matching_is_visible_in_trace():
    retriever = EvidenceRetriever()
    results = retriever.retrieve(
        "habitat and pollinators",
        top_k=10,
        metrics=["pollinator presence"],
        practices=["hedgerows"],
    )
    assert results
    assert any("pollinator presence" in result.filter_matches for result in results)


def test_empty_query_does_not_return_unrelated_evidence():
    assert EvidenceRetriever().retrieve("   ") == []
