"""
FastAPI entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000

Then POST to /verify with:
    {"disclosure": "...", "draft_claims_text": "..."}
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph import run_pipeline
from app.schemas import QualityReport

app = FastAPI(
    title="Patent Verification & Evaluation Agent System",
    description=(
        "Verifies AI-generated patent claims against inventor disclosures: "
        "grounds every claim element in evidence, flags hallucinations, and "
        "surfaces invention concepts the draft left out."
    ),
    version="0.1.0",
)


class VerifyRequest(BaseModel):
    disclosure: str
    draft_claims_text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify", response_model=QualityReport)
def verify(payload: VerifyRequest):
    if not payload.disclosure.strip() or not payload.draft_claims_text.strip():
        raise HTTPException(
            status_code=400, detail="Both disclosure and draft_claims_text are required."
        )
    final_state = run_pipeline(payload.disclosure, payload.draft_claims_text)
    if final_state.report is None:
        raise HTTPException(status_code=500, detail="Pipeline did not produce a report.")
    return final_state.report
