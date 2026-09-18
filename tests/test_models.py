from biosage.models import EnvironmentalProfile, EvidenceRecord


def test_profile_categories_require_measured_or_descriptive_conditions():
    profile = EnvironmentalProfile(region="somewhere", latitude=10, longitude=20)
    assert profile.provided_categories == set()
    assert not profile.is_sufficient_for_assessment

    profile = EnvironmentalProfile(
        land_use="cropland",
        soil_ph=6.5,
        rainfall_mm_year=700,
    )
    assert profile.provided_categories == {"land_use", "soil", "climate_water"}
    assert profile.is_sufficient_for_assessment


def test_profile_rejects_invalid_coordinates_and_ph():
    try:
        EnvironmentalProfile(latitude=100)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid latitude should be rejected")

    try:
        EnvironmentalProfile(soil_ph=15)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid pH should be rejected")


def test_evidence_tags_are_normalized():
    record = EvidenceRecord(
        evidence_id="E999",
        title="Example source",
        organization="FAO",
        year=2024,
        url="https://www.fao.org/",
        source_locator="section 1",
        passage="Paraphrase: A sufficiently long evidence passage for validation.",
        topics=[" Soil Health ", "soil health"],
        metrics=[" Soil Organic Carbon "],
        conditions=["Cropland"],
        practices=["Cover crops"],
        applicability=["farm fields"],
    )
    assert record.topics == ["soil health"]
    assert record.metrics == ["soil organic carbon"]
