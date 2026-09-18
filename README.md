# BioSage

BioSage is a Streamlit MVP for evidence-grounded biodiversity and farm-restoration planning. It provides typed environmental profiles, deterministic text/JSON normalization, targeted clarification, explainable rule-based recommendations, a local evidence corpus, traceable TF-IDF retrieval, bounded conversation memory, and lazy Gemini drafting with a deterministic fallback.

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
