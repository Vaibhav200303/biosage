"""Deterministic profile parsing for both JSON and conversational first turns."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import EnvironmentalProfile


ALIASES = {
    "ph": "soil_ph",
    "soil_ph": "soil_ph",
    "soil pH": "soil_ph",
    "soil ph": "soil_ph",
    "soc": "soil_organic_carbon_pct",
    "soil carbon": "soil_organic_carbon_pct",
    "soil organic carbon": "soil_organic_carbon_pct",
    "soil_organic_carbon": "soil_organic_carbon_pct",
    "rainfall": "rainfall_mm_year",
    "annual rainfall": "rainfall_mm_year",
    "temperature": "temperature_c_mean",
    "crop": "crop_system",
    "crops": "crop_system",
    "land use": "land_use",
    "land_use": "land_use",
    "water": "water_availability",
    "soil moisture": "soil_moisture",
    "pollution": "pollution_pressure",
    "fragmentation": "fragmentation_pressure",
    "pollinators": "pollinator_presence",
}


def _coerce_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group())
    return value


def _canonical_dict(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key).strip()
        canonical = ALIASES.get(key, ALIASES.get(key.lower(), key))
        if canonical == "rainfall_mm_year" and isinstance(value, str):
            qualitative = value.strip().lower()
            if qualitative in {"low", "scarce", "dry"}:
                canonical, value = "water_availability", "scarce"
            elif qualitative in {"high", "excess", "wet"}:
                canonical, value = "water_availability", "excess"
        if canonical in {"soil_ph", "soil_organic_carbon_pct", "rainfall_mm_year", "temperature_c_mean", "species_richness"}:
            value = _coerce_number(value)
        normalized[canonical] = value
    return normalized


def parse_profile_text(text: str) -> dict[str, Any]:
    """Extract only safe, deterministic fields from a natural-language turn."""

    result: dict[str, Any] = {}
    lowered = text.lower()
    patterns = {
        "soil_ph": r"(?:soil\s*)?p\s*h\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        "soil_organic_carbon_pct": r"(?:soc|soil\s+organic\s+carbon)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*%?",
        "rainfall_mm_year": r"(?:rainfall|rain fall|precipitation)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
        "temperature_c_mean": r"(?:temperature|temp)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)\s*(?:°?c)?",
        "species_richness": r"(?:species richness|species count)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, lowered)
        if match:
            result[field] = float(match.group(1))

    # Negation and explicit qualitative values win before broad positive keywords.
    pollution_low = re.search(r"(?:no|none|without|not)\s+(?:any\s+)?(?:pollution|contamination)|not\s+polluted|not\s+contaminated|unpolluted|pollution\s+(?:is\s+)?(?:low|none)|low\s+pollution", lowered)
    if pollution_low:
        result["pollution_pressure"] = "none" if any(token in pollution_low.group(0) for token in ("no", "none", "without", "not polluted", "not contaminated", "unpolluted")) else "low"
    elif re.search(r"pollution\s+(?:is\s+)?(?:high|heavy|severe)|\bheavily\s+polluted\b|\bpolluted\b|\bcontaminated\b|industrial\s+drain", lowered):
        result["pollution_pressure"] = "high"
    elif re.search(r"pollution\s+(?:is\s+)?moderate|moderate\s+pollution", lowered):
        result["pollution_pressure"] = "moderate"

    def qualitative_pressure(field: str, low_pattern: str, high_pattern: str) -> None:
        absence = re.search(rf"(?:no|none|without|not)\s+(?:any\s+)?{field.replace('_pressure', '')}", lowered)
        if absence:
            result[field] = "none"
            return
        low = re.search(low_pattern, lowered)
        high = re.search(high_pattern, lowered)
        if low:
            result[field] = "low"
        elif high:
            result[field] = "high"

    qualitative_pressure("deforestation_pressure", r"(?:no|none|without|not)\s+deforestation|deforestation\s+(?:is\s+)?(?:low|none)|low\s+deforestation", r"deforestation\s+(?:is\s+)?(?:high|heavy|severe)|high\s+deforestation|forest\s+loss\s+(?:is\s+)?high")
    qualitative_pressure("fragmentation_pressure", r"(?:no|none|without|not)\s+fragmentation|fragmentation\s+(?:is\s+)?(?:low|none)|low\s+fragmentation", r"fragmentation\s+(?:is\s+)?(?:high|heavy|severe)|high\s+fragmentation|fragmented\s+fields")
    qualitative_pressure("pollinator_presence", r"(?:no|few|low)\s+pollinators?|pollinators?\s+(?:are\s+)?(?:low|few)|pollinator\s+decline", r"(?:high|many)\s+pollinators?|pollinators?\s+(?:are\s+)?high")
    qualitative_pressure("habitat_diversity", r"habitat\s+diversity\s+(?:is\s+)?low|low\s+habitat\s+diversity", r"habitat\s+diversity\s+(?:is\s+)?high|high\s+habitat\s+diversity")
    qualitative_pressure("soil_moisture", r"soil\s+moisture\s+(?:is\s+)?low|low\s+soil\s+moisture", r"soil\s+moisture\s+(?:is\s+)?high|high\s+soil\s+moisture")
    if "waterlogged" in lowered:
        result["soil_moisture"] = "waterlogged"

    if re.search(r"(?:low rainfall|rainfall\s+(?:is\s+)?low|rainfall\s+scarce|water\s+scarce|water\s+scarcity|drought|dryland)", lowered):
        result["water_availability"] = "scarce"
    elif re.search(r"(?:high rainfall|rainfall\s+(?:is\s+)?high|wet|excess water)", lowered):
        result["water_availability"] = "excess"
    if any(token in lowered for token in ("monoculture", "continuous wheat", "single crop")):
        result["crop_system"] = text.strip()
    if any(token in lowered for token in ("cropland", "farm", "field", "orchard", "pasture")):
        result.setdefault("land_use", text.strip())
    if any(token in lowered for token in ("semi-arid", "semi arid", "dryland")):
        result.setdefault("region", "semi-arid site (user-described)")
    result["notes"] = text.strip()
    return result


def merge_profiles(base: EnvironmentalProfile | None, update: EnvironmentalProfile | dict[str, Any] | str) -> EnvironmentalProfile:
    """Merge a new JSON/text turn over prior values without inventing defaults."""

    if isinstance(update, EnvironmentalProfile):
        update_data = update.model_dump(exclude_none=True)
    elif isinstance(update, str):
        stripped = update.strip()
        try:
            parsed = json.loads(stripped) if stripped.startswith("{") else parse_profile_text(stripped)
        except json.JSONDecodeError:
            parsed = parse_profile_text(stripped)
        update_data = _canonical_dict(parsed)
    else:
        update_data = _canonical_dict(update)
    base_data = base.model_dump(exclude_none=True) if base else {}
    base_data.update({key: value for key, value in update_data.items() if value is not None and value != ""})
    return EnvironmentalProfile.model_validate(base_data)


def clarification_questions(profile: EnvironmentalProfile) -> list[str]:
    """Ask at most three targeted questions for missing assessment categories."""

    questions: list[str] = []
    categories = profile.provided_categories
    if "land_use" not in categories:
        questions.append("What is the current land use and crop or vegetation system?")
    if "soil" not in categories:
        questions.append("What do you know about soil pH, organic carbon, texture, or moisture?")
    if "climate_water" not in categories:
        questions.append("What are the typical rainfall or water-availability conditions?")
    if "biodiversity" not in categories and len(questions) < 3:
        questions.append("How would you describe habitat diversity, species richness, or pollinator presence?")
    if "human_pressure" not in categories and len(questions) < 3:
        questions.append("Are pollution, deforestation, or fragmentation pressures present?")
    return questions[:3]
