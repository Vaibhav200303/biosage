"""Streamlit entrypoint for the BioSage MVP.

Milestones 1-2 expose the profile and evidence layer. Reasoning and Gemini synthesis
are intentionally added in the next milestone without changing these contracts.
"""

from __future__ import annotations

import json

import streamlit as st

from biosage.config import get_settings
from biosage.models import EnvironmentalProfile
from biosage.profiles import sample_profiles
from biosage.reasoning import derive_retrieval_tags
from biosage.retrieval import get_retriever


st.set_page_config(page_title="BioSage", page_icon="🌱", layout="wide")
st.title("🌱 BioSage")
st.caption("Evidence-grounded biodiversity and land-restoration planning")

settings = get_settings()
presets = sample_profiles()
with st.sidebar:
    st.header("Demo profile")
    preset_name = st.selectbox("Start with a scenario", ["Custom", *presets])
    selected = presets.get(preset_name)
    if selected:
        st.info("A judge-ready scenario is loaded. Adjust values in the JSON editor below.")
    st.caption(f"Local evidence records are retrieved with TF-IDF. Top {settings.retrieval_top_k} shown.")

default_profile = selected.model_dump_json(indent=2) if selected else EnvironmentalProfile().model_dump_json(indent=2)
profile_json = st.text_area("Environmental profile (JSON)", value=default_profile, height=300)
query = st.text_input(
    "Evidence question",
    value="How can I improve soil health, water resilience, and biodiversity in this system?",
)

if st.button("Retrieve grounded evidence", type="primary"):
    try:
        profile = EnvironmentalProfile.model_validate(json.loads(profile_json))
        query_context = " ".join(filter(None, [query, profile.land_use, profile.crop_system, profile.region]))
        tags = derive_retrieval_tags(profile, query_context)
        results = get_retriever().retrieve(query_context, top_k=settings.retrieval_top_k, **tags, min_score=settings.retrieval_min_score)
        st.subheader("Profile coverage")
        st.write(f"Categories provided: {', '.join(sorted(profile.provided_categories)) or 'none'}")
        st.json(profile.model_dump(exclude_none=True))
        st.subheader("Evidence retrieved")
        if not results:
            st.warning("No evidence met the relevance threshold. Try a broader question.")
        for result in results:
            record = result.evidence
            with st.expander(f"{record.evidence_id} · {record.title} · score {result.score:.3f}"):
                st.write(record.passage)
                st.caption(f"Matched terms: {', '.join(result.matched_terms) or 'metadata only'}")
                st.caption(f"Score trace — lexical {result.lexical_score:.3f} · tags {result.tag_score:.3f} · applicability {result.applicability_score:.3f} · penalty {result.metadata_penalty:.2f}")
                if result.score_notes:
                    st.caption("; ".join(result.score_notes))
                st.caption(f"Source locator: {record.source_locator}")
                st.markdown(f"[{record.organization} ({record.year or 'n.d.'})]({record.url})")
    except Exception as exc:
        st.error(f"Please provide a valid profile JSON: {exc}")
