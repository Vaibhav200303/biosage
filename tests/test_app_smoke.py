from pathlib import Path


def test_streamlit_entrypoint_wires_all_user_modes_and_grounded_rendering():
    source = Path("app.py").read_text(encoding="utf-8")
    for expected in ("Chat", "Structured form", "JSON", "generate_assessment", "ConversationMemory", "Evidence retrieved and score trace", "st.map", "st.number_input", "soil_organic_carbon_pct", "pollution_pressure"):
        assert expected in source
