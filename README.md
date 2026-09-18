# BioSage

BioSage is an evidence-grounded biodiversity and farm-restoration assistant designed for the attached hackathon challenge and a 24-hour build. It turns a short site description into a transparent assessment: it checks whether enough environmental context is available, retrieves relevant scientific evidence, connects soil/water/land-use/biodiversity pressures, and returns practical recommendations with source links and measurable indicators.

The MVP is intentionally focused on decision support rather than automated diagnosis. It uses a curated local evidence corpus and a deterministic rules engine as its safety net. Gemini is optional and can polish a grounded seed response, but it cannot add recommendations, citations, confidence values, or unsupported numeric effects.

## Challenge alignment

BioSage demonstrates the challenge capabilities most valuable in a hackathon judging flow:

- Multi-metric reasoning across soil condition, water/climate, land use, biodiversity, and human pressure.
- Targeted clarification when fewer than three meaningful environmental categories are supplied.
- At least three actionable recommendations for a complete assessment.
- Mechanisms, implementation steps, time horizons, caveats, confidence components, and measurable indicators.
- Traceable evidence IDs, source locators, direct source URLs, retrieval scores, and score-penalty notes.
- Conversational follow-up memory with a six-turn bound.
- A constrained follow-up example: “What if I cannot stop wheat production?” changes the advice to an intercrop/rotation pilot while retaining the primary crop.
- A deterministic offline mode that continues to work without an API key or live web search.

## Key features

1. Three judge-ready presets: **Semi-arid monoculture**, **Acidic high-rainfall farm**, and **Polluted fragmented farmland**.
2. Three input modes: Chat, typed Structured form, and JSON.
3. Safe text/JSON normalization, aliases, profile merging, and negation-aware qualitative parsing.
4. Local TF-IDF retrieval with metadata tags for metrics, conditions, practices, and applicability.
5. Rule-specific intervention selection, including pollution safety/source control, soil cover, crop diversification, habitat connectivity, water resilience, and baseline monitoring.
6. Optional Gemini `GeminiDraft` synthesis with strict Pydantic validation, citation subset checks, numeric/effect-language rejection, retries, and a local timeout.
7. Optional coordinate map when latitude and longitude are supplied; coordinates never count as environmental evidence.

## Architecture

```text
Streamlit Chat / Structured form / JSON / preset
                         ↓
profile normalization + merge + six-turn session memory
                         ↓
readiness gate: three meaningful environmental categories
                         ↓
rule-specific TF-IDF + metadata retrieval
                         ↓
deterministic grounded seed assessment
                         ↓
optional GeminiDraft rewrite (fail closed)
                         ↓
recommendation cards + reasoning chain + evidence trace
```

Core modules:

- `biosage/models.py` — strict Pydantic contracts for profiles, evidence, recommendations, and responses; unique evidence and citation-subset validation.
- `biosage/normalization.py` — natural-language parsing, JSON aliases, qualitative values, negation precedence, merging, and clarification questions.
- `biosage/knowledge.py` — cached JSONL loading and corpus validation.
- `biosage/retrieval.py` — cached scikit-learn TF-IDF retrieval, dependency-free fallback, metadata scoring, and transparent penalties.
- `biosage/reasoning.py` — environmental tag derivation, intervention rules, reasoning chains, heuristic confidence, and offline fallback.
- `biosage/gemini.py` — lazy Google GenAI client, small draft schema, grounding checks, timeout/retry handling, and fallback orchestration.
- `biosage/memory.py` — bounded conversation turns and Streamlit session-state helper.
- `app.py` — Streamlit website entrypoint and result rendering.

## Data, evidence, and grounding

`data/knowledge.jsonl` contains 41 short paraphrased records from authoritative FAO, IPCC, USDA NRCS, UNEP, and IPBES sources. Each record includes:

- Stable ID such as `E041`.
- Title, organization, year when known, and direct URL.
- A source locator and a clearly labeled paraphrase.
- Topics, supported metrics, conditions, practices, and applicability tags.
- Optional structured numeric claims; the current corpus intentionally relies on directional evidence rather than invented effect percentages.

Retrieval exposes separate lexical, tag, and applicability components. If requested metadata does not match, a named penalty is applied and displayed. A default minimum score prevents zero-overlap records from appearing as relevant evidence. Rule-specific metadata is applied before broad profile tags so a land-use baseline is not crowded out by generic biodiversity text.

## Profiles, reasoning, and confidence

`EnvironmentalProfile` supports optional values for soil pH, soil organic carbon, moisture, texture, rainfall, temperature, water availability, land use, crop system, species richness, habitat diversity, pollinators, pollution, deforestation, fragmentation, coordinates, and notes.

The readiness gate requires at least three meaningful categories from land use, soil, climate/water, biodiversity, or human pressure. Region, latitude, longitude, and `unknown` values do not satisfy the gate. Incomplete inputs produce up to three targeted questions.

The deterministic engine selects only rules whose conditions match the profile. Each recommendation is grounded in retrieved evidence and contains:

- Action and ecological mechanism.
- Affected metrics and implementation steps.
- Time horizon, caveats, and measurable indicators.
- Evidence IDs and a heuristic confidence label/score.

Confidence is decision-support guidance, not a probability. It combines evidence support, profile completeness, context applicability, and source diversity. Low applicability, weak evidence, or single-source support caps the score. The reasoning chain names supplied profile fields and explains their interaction rather than presenting a generic list of environmental concepts.

## Gemini integration and offline fallback

Gemini is optional. `generate_assessment()` always builds and validates the deterministic local seed first. If `GEMINI_API_KEY` is available, the lazy client sends only the profile, canonical retrieved evidence, bounded recent memory, and seed. Gemini returns a small `GeminiDraft` containing a summary, reasoning prose, and copied evidence IDs.

The draft is rejected if it:

- Uses an evidence ID outside the supplied set or omits a seed citation.
- Contains digits, percentages, “fold/double/triple/twice/half” effect language, or malformed JSON/schema.
- Attempts to change recommendations, evidence records, confidence, or safety assumptions.

Quota/rate failures, malformed responses, missing SDK/key, and local timeout return the deterministic seed. A locally timed-out call is not retried while its SDK worker may still be running.

## Exact website usage flow

1. Start the app and choose a preset in the left sidebar, or leave **Custom** selected.
2. Choose an input mode:
   - **Chat:** type a site description or follow-up in the chat box and press Enter.
   - **Structured form:** fill the typed controls for land use, crop system, pH, SOC, rainfall, water/moisture, habitat, pollinators, pollution, fragmentation, and optional coordinates; add a question and click **Assess site**.
   - **JSON:** enter a complete profile or use aliases such as `soil ph`, `soil organic carbon`, and qualitative `rainfall: "low"`; add a question and click **Assess site**.
3. BioSage merges the new turn into the session profile and stores the user/assistant turn in the six-turn memory.
4. If context is incomplete, answer the displayed clarification questions in Chat or Structured form and submit again.
5. For a complete response, review the reasoning chain, assumptions, recommendation cards, confidence components, caveats, and indicators.
6. Expand **Evidence retrieved and score trace** to inspect every cited passage, direct source link, source locator, lexical score, tag score, applicability score, metadata penalty, and score note.
7. Use **Reset conversation** before repeating a demo with another preset. Changing a preset also resets its profile and memory.

## Presets and demo script

### Demo 1 — Semi-arid monoculture

Choose **Semi-arid monoculture** and submit the default question. Show soil cover, crop diversification, habitat, and water-resilience recommendations. Then ask in Chat: `What if I cannot stop wheat production?` The diversification recommendation should preserve wheat and propose a compatible pilot rather than telling the user to stop or replace it.

### Demo 2 — Acidic high-rainfall farm

Choose **Acidic high-rainfall farm**. Highlight pH/soil baseline, climate-water context, bare-fallow cover advice, and the absence of pollution-source-control advice when pollution pressure is low.

### Demo 3 — Polluted fragmented farmland

Choose **Polluted fragmented farmland**. Highlight testing and source control before expanding food production, habitat connectivity, low pollinator context, fragmentation reasoning, and direct pollution/water-quality sources.

## Setup

Python 3.12 or newer is required.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m streamlit run app.py
```

### Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m streamlit run app.py
```

Gemini is optional. Configure it through an environment variable:

```bash
export GEMINI_API_KEY="your-key"
```

PowerShell equivalent:

```powershell
$env:GEMINI_API_KEY = "your-key"
```

Or copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and add the key. The real secrets file is ignored by Git. Optional settings include `BIOSAGE_GEMINI_MODEL`, `BIOSAGE_RETRIEVAL_TOP_K`, and `BIOSAGE_RETRIEVAL_MIN_SCORE`.

## Tests and CI

Run the local checks:

```bash
pytest -q
python -m compileall -q biosage app.py
```

The suite covers profile normalization/aliases, negation and qualitative parsing, readiness, corpus validation, TF-IDF and forced fallback retrieval, zero-overlap filtering, rule grounding, confidence/citation integrity, Gemini malformed/numeric fallback, memory bounds, the constrained crop follow-up, and Streamlit AppTest reset behavior. GitHub Actions in `.github/workflows/ci.yml` runs compilation and tests on Python 3.12 for pushes and pull requests. The current dependency-enabled local result is **32 passed**, including the unskipped AppTest; compilation, Ruff, and diff checks also pass. The available local virtual environment reports Python 3.14.6, so a clean Python 3.12 verification remains part of deployment validation.

## Streamlit Cloud deployment

1. Push the repository to a public or reviewer-accessible GitHub repository.
2. In Streamlit Community Cloud, create an app from branch **`master`** with entrypoint **`app.py`**.
3. In **Advanced settings**, choose **Python 3.12**. `runtime.txt` is not needed.
4. Community Cloud installs the root **`requirements.txt`**. `pyproject.toml` is used for local editable/dev installation and CI.
5. Add `GEMINI_API_KEY = "..."` under the app’s Secrets settings if Gemini drafting is desired. The key is optional; offline deterministic mode works without it.
6. Open the deployed URL and run all three preset demos from a clean browser session.

## Repository structure

```text
.
├── app.py
├── biosage/
│   ├── config.py
│   ├── gemini.py
│   ├── knowledge.py
│   ├── memory.py
│   ├── models.py
│   ├── normalization.py
│   ├── profiles.py
│   ├── reasoning.py
│   └── retrieval.py
├── data/knowledge.jsonl
├── tests/
├── .github/workflows/ci.yml
├── .streamlit/secrets.example.toml
├── pyproject.toml
└── requirements.txt
```

## Limitations and security/privacy

- This is a hackathon MVP, not agronomic, environmental, food-safety, or legal advice.
- The local corpus is curated and static; there is no live web research, remote sensing, weather feed, soil database, persistent database, user authentication, or automatic lab integration.
- Numeric effects are not estimated when the local evidence does not provide a compatible structured claim.
- Coordinates are displayed only as an optional map point and are not used as proof of climate or soil conditions.
- Session memory lives in Streamlit session state and is bounded to six turns; it is not a durable user record.
- Do not enter personally identifying, confidential farm, or regulated contamination data into the demo. If Gemini is enabled, submitted profile/query context and supplied evidence are sent to the configured Gemini API according to that provider’s terms.
- Keep API keys in environment variables or Streamlit secrets. Never commit `.streamlit/secrets.toml`, `.env`, or credentials to the repository.

## Submission status

The local implementation, evidence corpus, tests, CI workflow, typed UI, offline fallback, and deployment documentation are complete. Current dependency-enabled local verification is **32 passed**, including AppTest, with compilation and Ruff passing. Repository and live-demo URLs are not yet published in this workspace. Remaining submission tasks are: verify in a clean Python 3.12 environment, publish the repository, deploy and verify the Streamlit URL, and attach the final challenge-required DOCX with repository and demo links.
