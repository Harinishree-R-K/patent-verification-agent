Orchestrated with LangGraph (`app/graph.py`). Steps [2]→[3] and [4] run as
parallel branches off claim extraction and join at scoring.

## Why retrieval has no LLM call

Evidence retrieval (BM25 + vector) is pure, local, and cheap. Verification —
checking if a claim element is actually entailed by the retrieved evidence —
is the one step that needs real language understanding, so that's the only
per-element LLM call in the pipeline. In a cost-sensitive production
deployment, `app/agents/verification.py` is the one file you'd swap to use a
local NLI (entailment) model instead of an API call, since it's fundamentally
an entailment-checking task, not open-ended generation.

## Setup

```bash
pip install -r requirements.txt
```

Set your LLM credentials by copying `.env.example` to `.env` and filling in your key:

```bash
cp .env.example .env
```

Recommended for testing — **Gemini (free, no credit card)**:
```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-latest
GEMINI_API_KEY=your-real-key-here
```
Get a free key at https://aistudio.google.com/apikey. Note: on the free tier,
Google may use your inputs/outputs to improve their models — fine for testing
with sample data, but don't run real confidential disclosures through it.
Use `gemini-flash-latest` rather than a pinned version number (e.g.
`gemini-2.5-flash`) — Google retires specific Flash versions periodically,
and the `-latest` alias always points at their current recommended model.

To use Anthropic instead (paid, cheap, higher quality):
```dotenv
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

To use OpenAI instead:
```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-your-real-key-here
```
(uncomment `openai` in `requirements.txt` and `pip install -r requirements.txt` again)

**Never commit your real `.env` file or paste your API key anywhere outside
it** — it's already excluded via `.gitignore`. Treat API keys like passwords:
if one is ever exposed (pasted in a chat, committed to git, etc.), rotate it
immediately at the provider's dashboard.

## Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

Opens an interactive UI: paste a disclosure and draft claims, click
"Run Verification", get a coverage score, hallucination count, and a
claim-by-claim ledger with cited evidence.

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d @sample_data/sample_request.json
```

(Or POST `{"disclosure": "...", "draft_claims_text": "..."}` directly.)

## Run the tests

```bash
pytest tests/ -v
```

`tests/test_pipeline_dry_run.py` mocks the LLM calls so the full pipeline —
extraction, retrieval, verification, coverage, scoring, all the state passing
through LangGraph — can be validated without an API key. It does NOT validate
prompt quality against a real model; do that with your own key before relying
on this for anything real.

## What's a stand-in vs. production-ready

Built to be honest about what's fully real vs. a deliberate simplification:

- **Real, production-shaped:** agent separation, Pydantic schemas, LangGraph
  orchestration with parallel branches, FastAPI + Streamlit, BM25 retrieval,
  provider-swappable LLM client.
- **Deliberate simplification:** the "vector" half of hybrid search uses
  TF-IDF + cosine similarity rather than a dense embedding model, so the
  whole thing runs offline with no model downloads. Swapping in real
  embeddings (ChromaDB/FAISS + sentence-transformers or an API embedding
  model) only requires changing `TfidfVectorIndex` in
  `app/retrieval/hybrid_search.py` — the interface (`.query(text)`) stays
  the same.
- **Sentence-level chunking** is used for the disclosure rather than
  semantic/paragraph-aware chunking, which would matter more on long,
  multi-page real disclosures.

