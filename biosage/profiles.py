"""Judge-ready sample site profiles used by the Streamlit demo."""

from __future__ import annotations

from copy import deepcopy

from .models import EnvironmentalProfile


SAMPLE_PROFILES: dict[str, EnvironmentalProfile] = {
    "Semi-arid monoculture": EnvironmentalProfile(
        region="Deccan semi-arid plateau",
        soil_ph=7.4,
        soil_organic_carbon_pct=0.3,
        soil_moisture="low",
        soil_texture="sandy",
        rainfall_mm_year=540,
        water_availability="scarce",
        land_use="irrigated cropland",
        crop_system="continuous wheat monoculture",
        species_richness=4,
        habitat_diversity="low",
        pollinator_presence="low",
        pollution_pressure="low",
        fragmentation_pressure="moderate",
    ),
    "Acidic high-rainfall farm": EnvironmentalProfile(
        region="humid high-rainfall upland",
        soil_ph=5.1,
        soil_organic_carbon_pct=1.0,
        soil_moisture="high",
        soil_texture="clayey",
        rainfall_mm_year=1800,
        water_availability="excess",
        land_use="rainfed cropland",
        crop_system="maize with bare fallow",
        species_richness=8,
        habitat_diversity="moderate",
        pollinator_presence="moderate",
        pollution_pressure="low",
        fragmentation_pressure="low",
    ),
    "Polluted fragmented farmland": EnvironmentalProfile(
        region="peri-urban agricultural fringe",
        soil_ph=6.8,
        soil_organic_carbon_pct=0.7,
        soil_moisture="moderate",
        soil_texture="loamy",
        rainfall_mm_year=900,
        water_availability="seasonal",
        land_use="small fragmented vegetable plots",
        crop_system="mixed vegetables near an industrial drain",
        species_richness=3,
        habitat_diversity="low",
        pollinator_presence="low",
        pollution_pressure="high",
        fragmentation_pressure="high",
    ),
}


def sample_profiles() -> dict[str, EnvironmentalProfile]:
    """Return copies so callers can edit a preset without mutating global state."""

    return {name: deepcopy(profile) for name, profile in SAMPLE_PROFILES.items()}
