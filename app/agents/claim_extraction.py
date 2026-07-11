"""
Agent 1: Claim Extraction Agent

Takes raw patent claim text (one or more claims, free text) and breaks it
into atomic claim elements — the granularity at which we can independently
verify "is this specific technical detail supported by the disclosure?"
"""
from __future__ import annotations
from typing import List

from app.llm_client import call_json
from app.schemas import ClaimElement

SYSTEM_PROMPT = """You are a patent claim parsing engine. Given raw patent \
claim text, break each claim into its atomic technical elements — the \
individual components, mechanisms, or limitations a claim asserts. \
Each element should be independently checkable against a disclosure \
(i.e. a specific technical assertion, not a whole sentence bundling \
several ideas). Preserve the original wording as closely as possible; \
do not paraphrase away technical specificity."""

JSON_SCHEMA_INSTRUCTIONS = """Return JSON with this exact shape:
{"elements": [{"claim_number": 1, "text": "..."}, ...]}"""


def extract_claim_elements(draft_claims_text: str) -> List[ClaimElement]:
    user_prompt = (
        f"{JSON_SCHEMA_INSTRUCTIONS}\n\nPatent claim text:\n{draft_claims_text}"
    )
    result = call_json(SYSTEM_PROMPT, user_prompt)
    elements = result.get("elements", [])

    claim_elements: List[ClaimElement] = []
    for i, el in enumerate(elements):
        claim_elements.append(
            ClaimElement(
                element_id=f"el_{i+1:03d}",
                text=el.get("text", "").strip(),
                claim_number=el.get("claim_number"),
            )
        )
    return claim_elements
