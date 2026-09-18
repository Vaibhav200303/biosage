import json

import pytest

from biosage.gemini import (
    GeminiGenerationError,
    GeminiSynthesizer,
    generate_assessment,
    validate_grounding,
)
from biosage.memory import ConversationMemory
from biosage.models import (
    AssessmentResponse,
    EnvironmentalProfile,
    EvidenceRecord,
    Recommendation,
    RetrievedEvidence,
)
from biosage.normalization import clarification_questions, merge_profiles
from biosage.profiles import sample_profiles
from biosage.reasoning import assess, derive_retrieval_tags
from biosage.retrieval import EvidenceRetriever


def test_json_and_text_profile_merge_without_invented_defaults():
    profile = merge_profiles(None, '{"land_use": "cropland", "soil_ph": 6.4}')
    profile = merge_profiles(profile, "Continuous wheat monoculture, SOC is 0.3%, rainfall is 500 mm, water scarce")
    assert profile.land_use == "cropland"
    assert profile.soil_ph == 6.4
    assert profile.soil_organic_carbon_pct == 0.3
    assert profile.rainfall_mm_year == 500
    assert profile.water_availability == "scarce"
    assert profile.latitude is None


def test_json_aliases_and_qualitative_rainfall_match_text():
    json_profile = merge_profiles(None, {"land use": "cropland", "soil organic carbon": 0.3, "soil ph": 5.2, "rainfall": "low", "soil moisture": "low"})
    text_profile = merge_profiles(None, "cropland, soil organic carbon is 0.3%, soil pH is 5.2, rainfall is low, soil moisture low")
    assert json_profile.soil_organic_carbon_pct == text_profile.soil_organic_carbon_pct == 0.3
    assert json_profile.soil_ph == text_profile.soil_ph == 5.2
    assert json_profile.water_availability == text_profile.water_availability == "scarce"
    assert json_profile.soil_moisture == text_profile.soil_moisture == "low"


@pytest.mark.parametrize(
    ("text", "field", "value"),
    [
        ("no pollution and no deforestation", "pollution_pressure", "none"),
        ("not contaminated and no deforestation", "pollution_pressure", "none"),
        ("pollution low, deforestation low", "pollution_pressure", "low"),
        ("soil moisture low and habitat diversity low", "soil_moisture", "low"),
        ("pollinators high and rainfall high", "pollinator_presence", "high"),
        ("deforestation high", "deforestation_pressure", "high"),
    ],
)
def test_qualitative_parser_handles_negation_and_order(text, field, value):
    profile = merge_profiles(None, text)
    assert getattr(profile, field) == value
    if "rainfall" in text:
        assert profile.water_availability == "excess"


def test_negated_pollution_does_not_trigger_high_pressure():
    profile = merge_profiles(None, "cropland with no pollution, soil pH 6, rainfall 800 mm")
    assert profile.pollution_pressure == "none"
    assert not any("contamination source" in item.action for item in assess(merge_profiles(profile, "soil organic carbon 0.8%, species richness 10" )).recommendations)


def test_unknown_values_do_not_satisfy_readiness():
    profile = EnvironmentalProfile(
        soil_texture="unknown",
        pollinator_presence="unknown",
        pollution_pressure="unknown",
        deforestation_pressure="unknown",
        fragmentation_pressure="unknown",
    )
    assert profile.provided_categories == set()
    assert len(clarification_questions(profile)) == 3


def test_deterministic_assessment_is_complete_and_grounded_for_presets():
    for profile in sample_profiles().values():
        response = assess(profile)
        assert response.status == "complete"
        assert len(response.recommendations) >= 3
        available = {item.evidence_id for item in response.evidence}
        assert available
        assert all(set(item.evidence_ids) <= available for item in response.recommendations)
        assert len(response.reasoning_variables) >= 3

    acidic = assess(sample_profiles()["Acidic high-rainfall farm"])
    assert not any("contamination source" in item.action for item in acidic.recommendations)
    polluted = assess(sample_profiles()["Polluted fragmented farmland"])
    assert {"pollution_pressure", "fragmentation_pressure", "species_richness"} <= set(polluted.reasoning_variables)
    land_use = next(item for item in acidic.recommendations if "land-use" in item.action)
    assert set(land_use.evidence_ids) <= {"E040", "E008", "E031"}


def test_multi_turn_primary_crop_constraint_adapts_diversification_advice():
    profile = sample_profiles()["Semi-arid monoculture"]
    response = assess(profile, query="What if I cannot stop wheat production?")
    diversification = next(item for item in response.recommendations if "wheat" in item.action.lower() or "intercrop" in item.action.lower())
    assert "keep" in diversification.action.lower()
    assert "replace" not in diversification.action.lower()
    assert "stop" not in diversification.action.lower()
    assert any("pilot" in step.lower() for step in diversification.implementation_steps)

    maize = profile.model_copy(update={"crop_system": "continuous maize"})
    maize_response = assess(maize, query="What if I cannot stop maize production?")
    maize_diversification = next(item for item in maize_response.recommendations if "crop" in item.action.lower() and "pilot" in item.action.lower())
    assert "wheat" not in maize_diversification.action.lower()

    replace_response = assess(profile, query="I must replace wheat next season")
    replace_diversification = next(item for item in replace_response.recommendations if "continuous monoculture" in item.action.lower())
    assert "keep the primary crop" not in replace_diversification.action.lower()


def test_tags_are_derived_from_pollution_profile_not_hardcoded_soil_defaults():
    tags = derive_retrieval_tags(sample_profiles()["Polluted fragmented farmland"], "contamination near drain")
    assert "pollution risk" in tags["metrics"]
    assert "water quality" in tags["metrics"]
    assert "soil organic carbon" in tags["metrics"]


def test_forced_dependency_free_fallback_has_nonzero_lexical_score(monkeypatch):
    from biosage import retrieval

    monkeypatch.setattr(retrieval, "TfidfVectorizer", None)
    monkeypatch.setattr(retrieval, "cosine_similarity", None)
    retriever = retrieval.EvidenceRetriever()
    results = retriever.retrieve("pollinators habitat", top_k=3)
    assert results
    assert max(result.lexical_score for result in results) > 0


def test_empty_corpus_and_zero_overlap_fail_closed():
    with pytest.raises(ValueError):
        EvidenceRetriever(records=[])
    retriever = EvidenceRetriever()
    assert retriever.retrieve("zzzz qqqq nonsense", top_k=5) == []


def _retrieved(evidence_id: str) -> RetrievedEvidence:
    record = EvidenceRecord(
        evidence_id=evidence_id,
        title="Test evidence",
        organization="FAO",
        year=None,
        url="https://www.fao.org/",
        source_locator="test section",
        passage="Paraphrase: Test evidence passage that is long enough for the schema.",
        metrics=["soil health"],
    )
    return RetrievedEvidence(evidence=record, score=0.8, lexical_score=0.8, tag_score=0.5, applicability_score=0.2)


def test_unknown_gemini_citation_is_rejected():
    evidence = _retrieved("E001")
    response = AssessmentResponse.model_construct(
        status="needs_clarification",
        clarifying_questions=["What is the soil condition?"],
        profile_summary="",
        assumptions=[],
        recommendations=[],
        reasoning_chain=[],
        reasoning_variables=[],
        evidence=[evidence],
    )
    response.evidence = [evidence]
    response.recommendations = [Recommendation.model_construct(evidence_ids=["E999"])]
    with pytest.raises(GeminiGenerationError):
        validate_grounding(response, {"E001"})


def test_duplicate_gemini_evidence_is_rejected():
    evidence = _retrieved("E001")
    response = AssessmentResponse.model_construct(
        status="needs_clarification",
        clarifying_questions=["What is the soil condition?"],
        profile_summary="",
        assumptions=[],
        recommendations=[],
        reasoning_chain=[],
        reasoning_variables=[],
        evidence=[evidence, evidence],
    )
    with pytest.raises(GeminiGenerationError):
        validate_grounding(response, {"E001"})


def test_generation_readiness_gate_precedes_injected_gemini():
    class MustNotRun:
        def generate(self, *args, **kwargs):
            raise AssertionError("Gemini must not run before clarification")

    response = generate_assessment(EnvironmentalProfile(region="only a place"), synthesizer=MustNotRun())
    assert response.status == "needs_clarification"


def test_generation_malformed_or_unavailable_gemini_uses_grounded_seed():
    class Broken:
        def generate(self, *args, **kwargs):
            raise GeminiGenerationError("malformed JSON")

    response = generate_assessment(sample_profiles()["Semi-arid monoculture"], synthesizer=Broken())
    assert response.status == "complete"
    assert len(response.recommendations) >= 3
    assert response.evidence


def test_injected_gemini_only_rewrites_small_draft_and_preserves_grounding():
    seed = assess(sample_profiles()["Semi-arid monoculture"])
    cited = sorted({evidence_id for recommendation in seed.recommendations for evidence_id in recommendation.evidence_ids})
    payload = {
        "profile_summary": "Gemini-polished summary",
        "assumptions": ["Draft is grounded in the local corpus."],
        "reasoning_chain": ["Input interaction", "Mechanism", "Measured review"],
        "evidence_ids": cited,
    }

    class Models:
        def generate_content(self, **kwargs):
            class Response:
                pass

            response = Response()
            response.text = json.dumps(payload)
            return response

    class Client:
        models = Models()

    result = generate_assessment(sample_profiles()["Semi-arid monoculture"], synthesizer=GeminiSynthesizer(client=Client(), timeout_seconds=2))
    assert result.profile_summary == "Gemini-polished summary"
    assert result.recommendations == seed.recommendations
    assert result.evidence == seed.evidence
    assert "Confidence is a heuristic decision-support score, not a probability." in result.assumptions


def test_gemini_numeric_effect_language_is_rejected_and_falls_back():
    seed = assess(sample_profiles()["Semi-arid monoculture"])

    class Models:
        def generate_content(self, **kwargs):
            class Response:
                pass

            response = Response()
            response.text = json.dumps({
                "profile_summary": "A 99% gain in one year",
                "assumptions": [],
                "reasoning_chain": ["1", "2", "3"],
                "evidence_ids": sorted({evidence_id for recommendation in seed.recommendations for evidence_id in recommendation.evidence_ids}),
            })
            return response

    class Client:
        models = Models()

    result = generate_assessment(sample_profiles()["Semi-arid monoculture"], synthesizer=GeminiSynthesizer(client=Client(), timeout_seconds=2))
    assert result.profile_summary == seed.profile_summary
    assert result.assumptions == seed.assumptions


def test_memory_is_bounded_to_six_turns():
    memory = ConversationMemory(max_turns=6)
    for index in range(8):
        memory.add("user" if index % 2 == 0 else "assistant", f"turn {index}")
    assert [turn.content for turn in memory.recent()] == [f"turn {index}" for index in range(2, 8)]
    assert all(item["role"] in {"user", "assistant"} for item in memory.as_prompt())
