# BioSage

BioSage is a Streamlit MVP for evidence-grounded biodiversity and farm-restoration planning. The current foundation provides typed environmental profiles, judge-ready presets, a local 40-record evidence corpus, and a traceable TF-IDF retriever. Gemini synthesis and deterministic recommendations are added in the next milestone.

## Run locally

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
streamlit run app.py
```

`GEMINI_API_KEY` is optional for the current milestone. When used later, set it as an environment variable or as `GEMINI_API_KEY` in Streamlit secrets. Never commit secrets.

## Evidence design

`data/knowledge.jsonl` contains short paraphrases, stable evidence IDs, direct source URLs, source locators, topics, applicable metrics, conditions, practices, and applicability tags. Numeric claims are reserved for structured source-supported claims with units, time horizons, and locators; the current corpus intentionally uses directional evidence only.

## Tests

```powershell
pytest -q
```
