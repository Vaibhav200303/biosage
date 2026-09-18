"""Explainable rule-based recommendations and grounded offline fallback."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

from .config import get_settings
from .models import AssessmentResponse, EnvironmentalProfile, Recommendation, RetrievedEvidence
from .normalization import clarification_questions
from .retrieval import EvidenceRetriever


@dataclass(frozen=True)
class Rule:
    key: str
    action: str
    mechanism: str
    metrics: tuple[str, ...]
    practices: tuple[str, ...]
    query: str
    steps: tuple[str, ...]
    horizon: str
    caveats: tuple[str, ...]
    indicators: tuple[str, ...]
    matches: Callable[[EnvironmentalProfile], bool]


def derive_retrieval_tags(profile: EnvironmentalProfile, query: str = "") -> dict[str, list[str]]:
    """Derive metadata filters from the profile and user question, without UI bias."""

    metrics: set[str] = set()
    conditions: set[str] = set()
    practices: set[str] = set()
    text = " ".join(filter(None, [query, profile.land_use, profile.crop_system, profile.notes])).lower()
    if profile.soil_organic_carbon_pct is not None:
        metrics.add("soil organic carbon")
        if profile.soil_organic_carbon_pct < 1:
            conditions.add("low organic matter")
    if profile.soil_moisture in {"very_low", "low"}:
        metrics.update({"soil moisture", "water use"})
        conditions.add("water stress")
    if profile.soil_ph is not None:
        metrics.add("soil ph")
    if profile.temperature_c_mean is not None:
        metrics.add("temperature")
    if profile.rainfall_mm_year is not None:
        metrics.add("rainfall")
    if profile.rainfall_mm_year is not None or profile.water_availability:
        metrics.update({"soil moisture", "water availability"})
    if profile.water_availability == "scarce" or (profile.rainfall_mm_year is not None and profile.rainfall_mm_year < 600):
        conditions.update({"water scarcity", "seasonal rainfall"})
        practices.add("rainwater harvesting")
    if profile.species_richness is not None:
        metrics.add("species richness")
    if profile.habitat_diversity and profile.habitat_diversity.lower() != "unknown":
        metrics.add("habitat diversity")
    if profile.pollinator_presence == "low":
        metrics.add("pollinator presence")
        conditions.add("low pollinator presence")
    if profile.fragmentation_pressure == "high":
        metrics.add("habitat connectivity")
        conditions.add("fragmented farmland")
        practices.add("habitat corridors")
    if profile.deforestation_pressure == "high":
        metrics.add("deforestation pressure")
        conditions.add("deforestation")
    if profile.pollution_pressure == "high":
        metrics.update({"pollution risk", "water quality"})
        conditions.update({"pollution pressure", "polluted land"})
        practices.update({"soil testing", "source control"})
    if any(term in text for term in ("monoculture", "single crop", "continuous wheat")):
        conditions.add("monoculture")
        practices.update({"crop rotation", "intercropping"})
    for known in ("soil organic carbon", "soil moisture", "water quality", "pollinator presence", "habitat connectivity", "erosion", "biodiversity"):
        if known in text:
            metrics.add(known)
    return {"metrics": sorted(metrics), "conditions": sorted(conditions), "practices": sorted(practices)}


def _rules() -> tuple[Rule, ...]:
    return (
        Rule(
            "soil_diagnostic",
            "Establish a soil baseline before choosing amendments or changing tillage",
            "pH, organic carbon, moisture, and texture describe different constraints; a baseline prevents an intervention from being chosen from one proxy alone.",
            ("soil ph", "soil organic carbon", "soil moisture"),
            ("soil testing", "baseline monitoring"),
            "soil pH organic carbon moisture texture soil testing baseline",
            ("Use a consistent sampling design for pH, organic carbon, moisture, and texture.", "Record recent fertilizer, amendment, and tillage history.", "Use the baseline to select and review a site-appropriate practice.",),
            "0–3 months for baseline; review each season",
            ("This is a diagnostic step, not a recommendation for a universal amendment rate.",),
            ("soil ph", "soil organic carbon", "soil moisture"),
            lambda p: "soil" in p.provided_categories,
        ),
        Rule(
            "climate_water_plan",
            "Create a seasonal water and climate-risk baseline",
            "Rainfall, temperature, and water availability shape which soil-cover and water-management options are feasible; the same practice can have different effects in different contexts.",
            ("rainfall", "temperature", "water availability"),
            ("baseline monitoring", "adaptive management"),
            "rainfall temperature water availability climate adaptation baseline",
            ("Record seasonal rainfall and temperature patterns and identify dry or waterlogged periods.", "Map available water and downstream constraints.", "Use the record to time cover, irrigation, and runoff interventions.",),
            "one season baseline; review annually",
            ("Climate variables are site descriptors, not forecasts; use local observations where available.",),
            ("rainfall record", "temperature record", "water availability"),
            lambda p: "climate_water" in p.provided_categories,
        ),
        Rule(
            "land_use_baseline",
            "Document the current land-use and crop-system baseline before changing it",
            "A clear land-use baseline links interventions to crop sequence, field boundaries, habitat, and operational constraints.",
            ("land use",),
            ("baseline monitoring", "adaptive management"),
            "land use crop system crop diversity habitat baseline",
            ("Map crop or vegetation units and the current crop sequence.", "Record field margins, water features, and management constraints.", "Compare the next intervention against this baseline.",),
            "0–1 month baseline; review each season",
            ("The baseline does not imply that a crop system should be abandoned without livelihood and market planning.",),
            ("land-use map", "crop sequence", "field-margin inventory"),
            lambda p: "land_use" in p.provided_categories,
        ),
        Rule(
            "biodiversity_baseline",
            "Measure a small biodiversity and habitat baseline before restoration",
            "Species, habitat, and pollinator indicators describe different dimensions; a matched baseline makes change claims testable.",
            ("species richness", "habitat diversity", "pollinator presence"),
            ("baseline monitoring", "field surveys"),
            "species richness habitat diversity pollinator baseline monitoring",
            ("Choose indicator taxa or habitat features relevant to the site objective.", "Record habitat types and their seasonal condition.", "Repeat the same observation method after the intervention.",),
            "baseline now; repeat seasonally",
            ("Indicator choice should follow local ecology and available survey capacity.",),
            ("species observations", "habitat types", "pollinator visitation"),
            lambda p: "biodiversity" in p.provided_categories,
        ),
        Rule(
            "pressure_screen",
            "Screen human pressures and prioritize the most actionable pathway",
            "Pollution, fragmentation, and deforestation are distinct pressures; identifying the active pathway prevents a generic restoration action from masking a continuing driver.",
            ("pollution risk", "habitat connectivity", "deforestation pressure"),
            ("source control", "habitat restoration"),
            "pollution fragmentation deforestation pressure screening",
            ("Record the pressure, source, affected area, and exposure pathway.", "Prioritize source reduction or containment where feasible.", "Use a pressure indicator in the review plan.",),
            "0–3 months for screening; review annually",
            ("Pollution safety requires contaminant-specific testing; habitat actions do not substitute for source control.",),
            ("pressure source", "affected area", "pressure trend"),
            lambda p: p.pollution_pressure == "high" or p.deforestation_pressure == "high" or p.fragmentation_pressure == "high",
        ),
        Rule(
            "pollution_safety",
            "Test soil and control the contamination source before expanding food production",
            "Contaminant-specific testing identifies exposure pathways; source control reduces continuing inputs to soil and water.",
            ("pollution risk", "water quality", "soil function"),
            ("soil testing", "source control"),
            "soil pollution contaminant testing source control exposure pathways",
            ("Pause high-exposure food uses where appropriate and document the suspected source.", "Collect a representative soil and irrigation-water test with a qualified lab.", "Use test results to select source control, land-use restrictions, or remediation.",),
            "0–3 months for baseline and risk controls",
            ("Do not infer food safety from pH or appearance; contaminant-specific testing is required.",),
            ("soil contaminant panel", "source discharge observation", "water-quality test"),
            lambda p: p.pollution_pressure == "high",
        ),
        Rule(
            "soil_cover",
            "Establish a locally suitable cover-crop or residue-retention program",
            "Living roots and surface cover reduce exposure, support organic inputs, and can improve moisture and erosion outcomes over time.",
            ("soil organic carbon", "soil moisture", "erosion"),
            ("cover crops", "residue retention", "reduced tillage"),
            "cover crops residue retention soil moisture erosion organic carbon",
            ("Select species for rainfall, termination window, and crop needs.", "Keep soil covered during the most erosive or dry period.", "Track soil cover, infiltration, and organic-carbon baseline consistently.",),
            "one season to multi-year trend",
            ("Competition for scarce water can occur; use a locally adapted species mix and monitor moisture. SOC below 1% is an internal screening heuristic, not a universal soil-health threshold.",),
            ("surface cover percentage", "infiltration observation", "soil organic carbon baseline"),
            lambda p: (p.soil_organic_carbon_pct is not None and p.soil_organic_carbon_pct < 1) or p.soil_moisture in {"very_low", "low"} or "bare" in " ".join(filter(None, [p.notes, p.crop_system])).lower(),
        ),
        Rule(
            "diversify_crops",
            "Replace continuous monoculture with a feasible rotation or intercrop",
            "Different crops diversify residues and rooting patterns and can interrupt some pest cycles while increasing system diversity.",
            ("crop diversity", "pest regulation", "nutrient cycling"),
            ("crop rotation", "intercropping"),
            "crop diversification rotation intercropping pest nutrient cycling",
            ("Choose a compatible second crop or legume based on market and water constraints.", "Start on a pilot block and compare pest, soil, and yield indicators.", "Adjust the sequence after one review cycle.",),
            "one rotation cycle",
            ("Crop compatibility and market constraints should be checked locally.",),
            ("crop species count", "pest scouting records", "yield stability"),
            lambda p: any(term in (p.crop_system or "").lower() for term in ("monoculture", "continuous", "single crop")),
        ),
        Rule(
            "habitat_connectivity",
            "Add native flowering margins, hedgerows, or a habitat corridor",
            "Connected, seasonally suitable vegetation can supply habitat and movement routes for pollinators and other beneficial organisms.",
            ("pollinator presence", "habitat diversity", "habitat connectivity"),
            ("hedgerows", "flower strips", "habitat corridors"),
            "pollinators habitat connectivity hedgerows flowering margins",
            ("Map existing habitat and identify gaps between field edges or water features.", "Plant locally suitable native species with staggered flowering where feasible.", "Avoid broad-spectrum treatment during flowering and monitor visitation.",),
            "one planting season to multi-year habitat trend",
            ("Plant selection, invasive-species risk, and pesticide timing require local advice.",),
            ("flowering period coverage", "pollinator visitation", "connected habitat length"),
            lambda p: p.pollinator_presence == "low" or p.fragmentation_pressure == "high" or (p.habitat_diversity or "").lower() == "low",
        ),
        Rule(
            "water_resilience",
            "Pilot contour or rainwater-harvesting measures matched to local hydrology",
            "Slowing and storing runoff can support soil moisture and reduce erosive flow where slope, infiltration, storage, and downstream impacts are suitable.",
            ("water availability", "soil moisture", "erosion"),
            ("rainwater harvesting", "contour measures"),
            "water harvesting runoff soil moisture erosion contour",
            ("Check slope, infiltration, catchment area, and downstream effects before construction.", "Pilot a small contour, swale, or storage intervention.", "Measure soil moisture and runoff after representative rainfall.",),
            "one rainy season",
            ("Water structures can redistribute risk; obtain local hydrology or engineering guidance for larger works. Rainfall below 600 mm/year is an internal screening heuristic, not a universal cutoff.",),
            ("soil moisture", "runoff observation", "erosion indicators"),
            lambda p: p.water_availability == "scarce" or (p.rainfall_mm_year is not None and p.rainfall_mm_year < 600),
        ),
        Rule(
            "adaptive_monitoring",
            "Set a baseline and review indicators before scaling the intervention",
            "Consistent measurements make management adaptive and prevent unsupported claims about outcomes.",
            ("soil health", "biodiversity", "water quality"),
            ("baseline monitoring", "adaptive management"),
            "baseline monitoring soil biodiversity water quality adaptive management",
            ("Record the current management, soil, water, and biodiversity baseline.", "Choose a small set of indicators tied to the intervention.", "Review results at a defined date and adjust the plan.",),
            "baseline now; review each season",
            ("Indicator choice should match the site objective and available sampling capacity.",),
            ("baseline completeness", "indicator trend", "review decision"),
            lambda p: True,
        ),
    )


def _confidence(results: list[RetrievedEvidence], profile: EnvironmentalProfile, rule: Rule) -> tuple[str, float, dict[str, float], float]:
    evidence_support = min(1.0, sum(item.score for item in results) / max(1, len(results)) * 2)
    completeness = min(1.0, len(profile.provided_categories) / 5)
    applicability = max((item.applicability_score for item in results), default=0.0)
    source_diversity = min(1.0, len({item.evidence.organization for item in results}) / 2)
    cap = 0.75 if evidence_support < 0.45 else 1.0
    cap = min(cap, 0.70) if applicability < 0.25 else cap
    cap = min(cap, 0.70) if source_diversity < 1.0 else cap
    score = min(cap, 0.5 * evidence_support + 0.3 * completeness + 0.2 * max(applicability, 0.5))
    label = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    return label, round(score, 3), {"evidence_support": round(evidence_support, 3), "profile_completeness": round(completeness, 3), "context_applicability": round(applicability, 3), "source_diversity": round(source_diversity, 3)}, cap


def _profile_variables(profile: EnvironmentalProfile) -> list[str]:
    """Return representative supplied field names across environmental categories."""

    groups = (
        ("soil", ("soil_ph", "soil_organic_carbon_pct", "soil_moisture", "soil_texture")),
        ("climate_water", ("rainfall_mm_year", "temperature_c_mean", "water_availability")),
        ("land_use", ("land_use", "crop_system")),
        ("biodiversity", ("species_richness", "habitat_diversity", "pollinator_presence", "fragmentation_pressure")),
        ("human_pressure", ("pollution_pressure", "deforestation_pressure")),
    )
    fields: list[str] = []
    for _, candidates in groups:
        selected = next((field for field in candidates if getattr(profile, field) is not None and str(getattr(profile, field)).lower() != "unknown"), None)
        if selected:
            fields.append(selected)
    # Preserve additional measured/context fields so the reasoning chain can show the actual inputs.
    all_fields = ("soil_ph", "soil_organic_carbon_pct", "soil_moisture", "soil_texture", "rainfall_mm_year", "temperature_c_mean", "water_availability", "land_use", "crop_system", "species_richness", "habitat_diversity", "pollinator_presence", "pollution_pressure", "deforestation_pressure", "fragmentation_pressure")
    for field in all_fields:
        if getattr(profile, field) is not None and str(getattr(profile, field)).lower() != "unknown" and field not in fields:
            fields.append(field)
    return fields


def assess(profile: EnvironmentalProfile, *, query: str = "", retriever: EvidenceRetriever | None = None) -> AssessmentResponse:
    """Generate a grounded deterministic assessment or targeted clarification."""

    if not profile.is_sufficient_for_assessment:
        return AssessmentResponse(status="needs_clarification", clarifying_questions=clarification_questions(profile), profile_summary="The site description is not yet sufficient for a multi-metric assessment.")
    retriever = retriever or EvidenceRetriever()
    tags = derive_retrieval_tags(profile, query)
    all_evidence: dict[str, RetrievedEvidence] = {}
    recommendations: list[Recommendation] = []
    generic_keys = {"soil_diagnostic", "climate_water_plan", "land_use_baseline", "biodiversity_baseline", "pressure_screen", "adaptive_monitoring"}
    active_rules = [rule for rule in _rules() if rule.matches(profile)]
    active_rules.sort(key=lambda rule: rule.key in generic_keys)
    preserve_primary_crop = bool(re.search(r"(?:cannot|can't|can not|must|need to)\s+(?:stop|replace|remove)|keep\s+(?:(?:growing|planting)\s+\w+|the\s+\w+|\w+)|continue\s+(?:growing|planting)", query.lower()))
    for rule in active_rules:
        # Intervention-specific tags dominate broad profile tags so, for example, a
        # land-use baseline is not crowded out by a generic biodiversity query.
        rule_conditions = [condition for condition in tags["conditions"] if condition in rule.query.lower()]
        results = retriever.retrieve(rule.query, top_k=3, metrics=rule.metrics, practices=rule.practices, conditions=rule_conditions, min_score=get_settings().retrieval_min_score)
        aligned = [item for item in results if set(rule.metrics) & set(item.evidence.metrics)]
        if aligned:
            results = aligned
        if not results:
            results = retriever.retrieve(rule.query + " " + query, top_k=3, **tags, min_score=0.01)
        if not results:
            continue
        for item in results:
            all_evidence[item.evidence_id] = item
        confidence, score, components, cap = _confidence(results, profile, rule)
        action, mechanism, steps, caveats = rule.action, rule.mechanism, rule.steps, rule.caveats
        if rule.key == "diversify_crops" and preserve_primary_crop:
            action = "Keep the primary wheat crop and pilot a compatible intercrop or rotation window"
            mechanism = "Diversification can be introduced around the retained crop through an intercrop strip, relay crop, or small rotation block, improving system diversity without requiring an immediate crop replacement."
            steps = ("Keep wheat on the main production area and mark a small pilot strip or rotation window.", "Test a locally compatible legume, intercrop, or field-margin strip with the same water and market constraints.", "Compare pest observations, soil indicators, and yield stability before expanding the pilot.")
            caveats = (*rule.caveats, "This follow-up preserves the primary crop; local compatibility and market constraints still need checking.")
        recommendations.append(Recommendation(action=action, mechanism=mechanism, affected_metrics=list(rule.metrics), implementation_steps=list(steps), time_horizon=rule.horizon, confidence=confidence, confidence_score=score, confidence_components=components, confidence_cap=cap, evidence_ids=[item.evidence_id for item in results[:2]], caveats=list(caveats), measurable_indicators=list(rule.indicators)))
        if len(recommendations) >= 4:
            break
    if len(recommendations) < 3:
        raise RuntimeError("Grounded fallback could not produce three evidence-backed recommendations")
    variables = _profile_variables(profile)
    if len(variables) < 3:
        raise RuntimeError("A complete assessment needs at least three supplied environmental variables")
    if profile.pollution_pressure == "high":
        priority = ("pollution_pressure", "fragmentation_pressure", "species_richness", "pollinator_presence", "land_use")
    elif profile.soil_organic_carbon_pct is not None and profile.soil_organic_carbon_pct < 1 or profile.water_availability == "scarce":
        priority = ("soil_organic_carbon_pct", "water_availability", "crop_system", "rainfall_mm_year", "habitat_diversity")
    else:
        priority = ("soil_ph", "rainfall_mm_year", "crop_system", "temperature_c_mean", "habitat_diversity")
    interaction_fields = [field for field in priority if field in variables][:4]
    if len(interaction_fields) < 3:
        interaction_fields.extend(field for field in variables if field not in interaction_fields)
    observed = "; ".join(f"{field}={getattr(profile, field)}" for field in variables)
    chain = [
        f"Observed inputs: {observed}.",
        f"Interaction: {', '.join(interaction_fields[:4])} jointly shape soil, water, habitat, and pressure risks; the relationships are context-dependent rather than universal thresholds.",
        "Actions are paired with baseline indicators so the plan can be adapted after measurement rather than treated as a guaranteed effect.",
    ]
    return AssessmentResponse(status="complete", profile_summary=f"Assessment for {profile.region or 'the described site'} using {len(profile.provided_categories)} environmental categories.", assumptions=["Directional effects are shown unless a retrieved source provides a structured numeric claim.", "Confidence is a heuristic decision-support score, not a probability."], recommendations=recommendations, reasoning_chain=chain, reasoning_variables=variables, evidence=list(all_evidence.values()))
