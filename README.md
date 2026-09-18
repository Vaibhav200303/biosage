# BioSage

BioSage is an evidence-grounded biodiversity and farm-restoration assistant built for a 24-hour hackathon. It accepts natural language, structured JSON, or a reusable scenario preset; asks for missing context; retrieves local authoritative evidence; and produces multi-metric recommendations with citations. Gemini is an optional drafting layer over a deterministic grounded seed, so the app remains usable without an API key.

## Architecture

```text
Chat / form / JSON
        ↓
profile normalization + six-turn session memory
        ↓
readiness gate (three meaningful categories)
        ↓
rule-specific TF-IDF + metadata retrieval
        ↓
deterministic grounded assessment
        ↓
optional GeminiDraft rewrite (fail-closed)
        ↓
recommendation cards + reasoning + evidence trace
```

Key modules:

- `biosage/models.py`: strict Pydantic contracts and citation integrity checks.
- `biosage/normalization.py`: safe text/JSON aliases, qualitative values, negation precedence, and profile merge.
- `biosage/retrieval.py`: cached TF-IDF index, dependency-free fallback, metadata scoring, penalty trace.
- `biosage/reasoning.py`: category-aware intervention rules, grounded recommendations, confidence components, and offline fallback.
- `biosage/gemini.py`: lazy injectable Gemini client, bounded draft schema, timeout/retry handling, numeric/effect guard, and citation subset checks.
- `biosage/memory.py`: bounded six-turn session memory.
- `data/knowledge.jsonl`: 41 paraphrased evidence records with stable IDs, source locators, metrics, conditions, practices, and direct URLs.

## Schema and grounding

`EnvironmentalProfile` contains optional soil, climate/water, land-use, biodiversity, pressure, and coordinate fields. Coordinates and region labels never count as measured readiness categories. Full assessment requires three meaningful categories.

`AssessmentResponse` is either `needs_clarification` (up to three questions) or `complete` (at least three recommendations, three supplied reasoning variables, non-empty evidence, and unique traceable evidence IDs). Every recommendation citation must belong to the response evidence subset. Unsupported numeric effects are rejected; the corpus currently uses directional evidence only.

## Local setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
streamlit run app.py
```

Gemini is optional. Set `GEMINI_API_KEY` as an environment variable or copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and add the key. The secrets file is ignored by Git.

## Testing and CI

```powershell
pytest -q
python -m compileall -q biosage app.py
```

GitHub Actions runs Python 3.12 compilation and the complete test suite on pushes and pull requests.

## Streamlit deployment

Deploy `app.py` from the repository root on Streamlit Community Cloud. Configure `GEMINI_API_KEY` under the app’s Secrets settings; deployment without it intentionally uses offline deterministic mode. No database, authentication, upload pipeline, or live geospatial API is required for the MVP.

## Judge demo flow

1. Choose **Semi-arid monoculture** and run the structured form to show cover, diversification, habitat, and water recommendations.
2. Choose **Polluted fragmented farmland** to show source-control-first safety logic and pressure-aware reasoning.
3. Use Chat mode with an incomplete message to show targeted clarification, then add pH/SOC/rainfall/crop details and submit a follow-up.
4. Expand **Evidence retrieved and score trace** to show lexical, tag, applicability, metadata-penalty, source locator, and direct citation links.
