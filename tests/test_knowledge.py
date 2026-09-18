from biosage.knowledge import load_knowledge_base, validate_corpus


def test_corpus_loads_with_unique_traceable_records():
    records = load_knowledge_base()
    assert 35 <= len(records) <= 60
    validate_corpus(records)
    assert len({record.evidence_id for record in records}) == len(records)
    assert all(record.source_locator for record in records)
    assert all(str(record.url).startswith("https://") for record in records)


def test_corpus_has_required_source_families_and_metrics():
    records = load_knowledge_base()
    organizations = {record.organization for record in records}
    assert {"FAO", "IPCC", "USDA NRCS", "UNEP", "IPBES"} <= organizations
    metrics = {metric for record in records for metric in record.metrics}
    assert {"soil organic carbon", "species richness", "water quality"} <= metrics
