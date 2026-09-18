"""Streamlit entrypoint for the BioSage 24-hour hackathon MVP."""

from __future__ import annotations

import json

import streamlit as st

from biosage.config import get_settings
from biosage.gemini import generate_assessment
from biosage.memory import ConversationMemory, memory_from_session
from biosage.models import AssessmentResponse, EnvironmentalProfile
from biosage.normalization import merge_profiles
from biosage.profiles import sample_profiles


st.set_page_config(page_title="BioSage", page_icon="🌱", layout="wide")
settings = get_settings()
presets = sample_profiles()


def _preset_changed() -> None:
    selected = st.session_state.get("preset_name", "Custom")
    if selected == "Custom":
        profile = EnvironmentalProfile()
    else:
        profile = presets[selected]
    st.session_state["profile"] = profile
    st.session_state["profile_text"] = profile.model_dump_json(indent=2)
    st.session_state["last_response"] = None
    st.session_state["biosage_memory"] = ConversationMemory(max_turns=settings.max_memory_turns)


def _reset_conversation() -> None:
    """Reset state before the next rerun instantiates the preset widget.

    Streamlit forbids assigning a widget's keyed session-state value after that
    widget has been created in the current run. Using this as the button callback
    makes the assignment happen before the rerun, so the selectbox can safely
    initialize with ``Custom``.
    """

    profile = EnvironmentalProfile()
    st.session_state["preset_name"] = "Custom"
    st.session_state["profile"] = profile
    st.session_state["profile_text"] = profile.model_dump_json(indent=2)
    st.session_state["last_response"] = None
    st.session_state["biosage_memory"] = ConversationMemory(max_turns=settings.max_memory_turns)


if "preset_name" not in st.session_state:
    st.session_state["preset_name"] = "Custom"
if "profile" not in st.session_state:
    st.session_state["profile"] = EnvironmentalProfile()
if "profile_text" not in st.session_state:
    st.session_state["profile_text"] = st.session_state["profile"].model_dump_json(indent=2)
if "last_response" not in st.session_state:
    st.session_state["last_response"] = None
memory = memory_from_session(st.session_state, max_turns=settings.max_memory_turns)


def _submit(update_text: str, query: str) -> None:
    try:
        profile = merge_profiles(st.session_state.get("profile"), update_text)
        st.session_state["profile"] = profile
        st.session_state["profile_text"] = profile.model_dump_json(indent=2)
        memory.add("user", query or update_text)
        response = generate_assessment(profile, query=query, memory=memory.as_prompt())
        memory.add("assistant", response.profile_summary or response.status)
        st.session_state["last_response"] = response
    except Exception as exc:
        st.session_state["last_response"] = None
        st.error(f"BioSage could not parse this turn: {exc}")


st.title("🌱 BioSage")
st.caption("Evidence-grounded biodiversity and land-restoration planning")

with st.sidebar:
    st.header("Scenario")
    st.selectbox("Judge-ready preset", ["Custom", *presets], key="preset_name", on_change=_preset_changed)
    st.caption("Presets reset the active profile and six-turn memory so demos are repeatable.")
    st.button("Reset conversation", on_click=_reset_conversation)
    st.divider()
    st.caption("Generation mode")
    st.info("Gemini draft + grounded local seed" if settings.gemini_api_key else "Offline deterministic mode (no API key)")

mode = st.radio("Input mode", ["Chat", "Structured form", "JSON"], horizontal=True)
query = ""
submitted = None

if mode == "Chat":
    st.write("Describe the site, add missing measurements, or ask a follow-up question.")
    submitted = st.chat_input("Example: soil organic carbon is 0.3%, rainfall is low, and this is continuous wheat monoculture")
    if submitted:
        query = submitted
        _submit(submitted, query)
else:
    with st.form(f"biosage_{mode.lower().replace(' ', '_')}_form"):
        if mode == "Structured form":
            st.caption("Optional fields are merged with the selected preset or current conversation profile.")
            land_use = st.text_input("Land use", placeholder="rainfed cropland")
            crop_system = st.text_input("Crop or vegetation system", placeholder="maize with bare fallow")
            col1, col2 = st.columns(2)
            with col1:
                has_ph = st.checkbox("Provide soil pH")
                soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, disabled=not has_ph)
                has_soc = st.checkbox("Provide soil organic carbon")
                soil_soc = st.number_input("SOC (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1, disabled=not has_soc)
                has_rainfall = st.checkbox("Provide annual rainfall")
                rainfall = st.number_input("Rainfall (mm/year)", min_value=0.0, value=700.0, step=25.0, disabled=not has_rainfall)
                water_options = {"Not provided": None, "scarce": "scarce", "seasonal": "seasonal", "adequate": "adequate", "excess": "excess"}
                water = st.selectbox("Water availability", list(water_options))
                moisture_options = {"Not provided": None, "very_low": "very_low", "low": "low", "moderate": "moderate", "high": "high", "waterlogged": "waterlogged"}
                moisture = st.selectbox("Soil moisture", list(moisture_options))
            with col2:
                habitat = st.selectbox("Habitat diversity", ["Not provided", "low", "moderate", "high"])
                pollinators = st.selectbox("Pollinator presence", ["Not provided", "low", "moderate", "high"])
                pollution = st.selectbox("Pollution pressure", ["Not provided", "none", "low", "moderate", "high"])
                fragmentation = st.selectbox("Fragmentation pressure", ["Not provided", "none", "low", "moderate", "high"])
                has_coords = st.checkbox("Provide coordinates")
                latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01, disabled=not has_coords)
                longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01, disabled=not has_coords)
            query = st.text_input("Question", value="How can I improve biodiversity and soil resilience?")
            submitted = st.form_submit_button("Assess site", type="primary")
            if submitted:
                typed = {
                    "land_use": land_use or None,
                    "crop_system": crop_system or None,
                    "soil_ph": soil_ph if has_ph else None,
                    "soil_organic_carbon_pct": soil_soc if has_soc else None,
                    "rainfall_mm_year": rainfall if has_rainfall else None,
                    "water_availability": water_options[water],
                    "soil_moisture": moisture_options[moisture],
                    "habitat_diversity": None if habitat == "Not provided" else habitat,
                    "pollinator_presence": None if pollinators == "Not provided" else pollinators,
                    "pollution_pressure": None if pollution == "Not provided" else pollution,
                    "fragmentation_pressure": None if fragmentation == "Not provided" else fragmentation,
                    "latitude": latitude if has_coords else None,
                    "longitude": longitude if has_coords else None,
                }
                _submit(json.dumps(typed), query)
        else:
            st.text_area("Environmental profile JSON or JSON aliases", key="profile_text", height=300)
            query = st.text_input("Question", value="How can I improve biodiversity and soil resilience?")
            submitted = st.form_submit_button("Assess site", type="primary")
            if submitted:
                _submit(st.session_state.get("profile_text", "{}"), query)

response: AssessmentResponse | None = st.session_state.get("last_response")
profile: EnvironmentalProfile = st.session_state.get("profile", EnvironmentalProfile())

if response is not None:
    st.subheader("Site profile")
    left, right = st.columns([2, 1])
    with left:
        st.json(profile.model_dump(exclude_none=True))
    with right:
        st.metric("Profile categories", len(profile.provided_categories))
        st.caption(", ".join(sorted(profile.provided_categories)) or "No measured categories yet")
        if profile.latitude is not None and profile.longitude is not None:
            st.map({"lat": [profile.latitude], "lon": [profile.longitude]})

    if response.status == "needs_clarification":
        st.warning("A full assessment needs a little more context.")
        for question in response.clarifying_questions:
            st.write(f"• {question}")
    else:
        st.subheader("Assessment")
        st.write(response.profile_summary)
        if response.assumptions:
            with st.expander("Assumptions and safety notes", expanded=True):
                for assumption in response.assumptions:
                    st.write(f"• {assumption}")
        st.subheader("Reasoning chain")
        for step in response.reasoning_chain:
            st.write(f"→ {step}")
        st.caption("Supplied variables: " + ", ".join(response.reasoning_variables))

        st.subheader("Recommendations")
        for index, recommendation in enumerate(response.recommendations, 1):
            with st.container(border=True):
                st.markdown(f"**{index}. {recommendation.action}**")
                st.write(recommendation.mechanism)
                st.caption(f"Metrics: {', '.join(recommendation.affected_metrics)} · Horizon: {recommendation.time_horizon}")
                st.caption(f"Heuristic confidence: {recommendation.confidence} ({recommendation.confidence_score:.2f}, cap {recommendation.confidence_cap:.2f})")
                if recommendation.confidence_components:
                    st.json(recommendation.confidence_components)
                st.write("Implementation steps")
                for step in recommendation.implementation_steps:
                    st.write(f"• {step}")
                if recommendation.caveats:
                    st.warning("Caveats: " + " ".join(recommendation.caveats))
                st.write("Indicators: " + ", ".join(recommendation.measurable_indicators))
                st.caption("Evidence IDs: " + ", ".join(recommendation.evidence_ids))

        with st.expander("Evidence retrieved and score trace", expanded=False):
            for item in response.evidence:
                record = item.evidence
                st.markdown(f"**{record.evidence_id} · {record.title}**")
                st.write(record.passage)
                st.caption(f"Score {item.score:.3f} = lexical {item.lexical_score:.3f} · tags {item.tag_score:.3f} · applicability {item.applicability_score:.3f} · penalty {item.metadata_penalty:.2f}")
                if item.score_notes:
                    st.caption("; ".join(item.score_notes))
                year = record.year or "n.d."
                st.markdown(f"[{record.organization} ({year}) — {record.source_locator}]({record.url})")

with st.expander("Conversation memory", expanded=False):
    for turn in memory.recent():
        st.caption(f"{turn.role}: {turn.content}")
