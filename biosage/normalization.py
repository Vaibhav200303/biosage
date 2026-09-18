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
    "soc": "soil_organic_carbon_pct",
    "soil carbon": "soil_organic_carbon_pct",
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

    keyword_fields = {
        "soil_moisture": (("waterlogged",), "waterlogged"),
        "water_availability": (("water scarce", "water scarcity", "low rainfall", "rainfall low", "rainfall is low", "drought", "dryland"), "scarce"),
        "water_availability_excess": (("high rainfall", "wet", "excess water"), "excess"),
        "pollution_pressure": (("polluted", "pollution", "industrial drain", "contaminated"), "high"),
        "fragmentation_pressure": (("fragmented", "fragmentation", "isolated fields"), "high"),
        "pollinator_presence": (("few pollinators", "low pollinator", "pollinator decline"), "low"),
    }
    for field, (keywords, value) in keyword_fields.items():
        if any(keyword in lowered for keyword in keywords):
            result[field.removesuffix("_excess")] = value
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
