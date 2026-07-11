"""
Agent 4: Coverage Analysis Agent

The inverse check of verification. Verification asks "is everything the
claims say true?" Coverage analysis asks "did the claims leave anything
important out?" — invention concepts present in the disclosure that
never made it into any claim element at all.
"""
from __future__ import annotations
from typing import List

from app.llm_client import call_json
from app.schemas import ClaimElement, CoverageGap

SYSTEM_PROMPT = """You are a patent coverage analyst. Compare an inventor \
disclosure against a set of patent claim elements. Identify distinct \
technical concepts, components, or features described in the disclosure \
that are NOT reflected in any of the claim elements. Focus on \
substantive omissions (a missing component, mechanism, or capability), \
not stylistic differences in wording."""

JSON_SCHEMA_INSTRUCTIONS = """Return JSON with this exact shape:
{"missing_concepts": [{"concept": "short phrase", "disclosure_evidence": "supporting sentence from disclosure"}]}
Return at most 5 items. If nothing meaningful is missing, return an empty list."""


def analyze_coverage(
    disclosure: str, claim_elements: List[ClaimElement]
) -> List[CoverageGap]:
    claims_block = "\n".join(f"- {el.text}" for el in claim_elements)
    user_prompt = (
        f"{JSON_SCHEMA_INSTRUCTIONS}\n\n"
        f"Disclosure:\n{disclosure}\n\n"
        f"Claim elements:\n{claims_block}"
    )

    try:
        parsed = call_json(SYSTEM_PROMPT, user_prompt)
        raw_gaps = parsed.get("missing_concepts", [])
    except ValueError:
        raw_gaps = []

    return [
        CoverageGap(
            concept=g.get("concept", "").strip(),
            disclosure_evidence=g.get("disclosure_evidence"),
        )
        for g in raw_gaps
        if g.get("concept")
    ]
