# Patent Verification & Evaluation Agent System

Verifies AI-generated patent claims against inventor disclosures. Grounds
every claim element in retrieved evidence, flags unsupported ("hallucinated")
claim elements, and surfaces invention concepts the draft left out.

## Architecture

```
disclosure + draft claims
        |
        v
 [1] Claim Extraction Agent      -- LLM: breaks claims into atomic elements
        |
        +----------------------------+
        v                            v
 [2] Evidence Retrieval Agent    [4] Coverage Analysis Agent
     (BM25 + vector, local,          (LLM: what's in the disclosure
      no LLM call)                    but missing from the claims?)
        |                            |
        v                            |
 [3] Verification Agent (LLM)        |
     SUPPORTED / UNSUPPORTED         |
     per element                     |
        |                            |
        +------------+---------------+
                     v
          [5] Quality Scoring Agent
              (pure aggregation, no LLM)
                     |
                     v
            Quality Report (JSON)
```

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

Set your LLM credentials as environment variables:

```bash
# Default provider is Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export LLM_PROVIDER="anthropic"          # or "openai" / "gemini"
export LLM_MODEL="claude-sonnet-4-6"     # or "gpt-4o", "gemini-1.5-pro", etc.
```

To use OpenAI instead: `export LLM_PROVIDER=openai`, `export OPENAI_API_KEY=...`,
and uncomment `openai` in `requirements.txt`. Same pattern for Gemini and Groq.

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

## Project layout

```
app/
  schemas.py              # Pydantic models / LangGraph state
  llm_client.py            # swappable Anthropic/OpenAI/Gemini wrapper
  graph.py                 # LangGraph orchestration
  main.py                  # FastAPI app
  agents/
    claim_extraction.py     # Agent 1
    evidence_retrieval.py   # Agent 2
    verification.py         # Agent 3
    coverage_analysis.py    # Agent 4
    quality_scoring.py       # Agent 5
  retrieval/
    hybrid_search.py         # BM25 + vector hybrid retriever
streamlit_app.py            # frontend
sample_data/                # example disclosure + draft claims
tests/
  test_pipeline_dry_run.py  # mocked end-to-end test
```
