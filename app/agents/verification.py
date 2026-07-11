"""
Agent 3: Verification Agent

For each claim element + its retrieved evidence passage, decides whether
the element is SUPPORTED or UNSUPPORTED by the disclosure, with a short
grounded reason. This is the one place per element that benefits from
real language understanding (paraphrase, implication, negation) rather
than keyword overlap alone.

Note on production architecture: in a cost-sensitive deployment, this
call can be swapped for a local NLI (natural language inference) model
(e.g. a DeBERTa entailment classifier) instead of an LLM API call per
element, since this is fundamentally an entailment-checking task. That
swap only touches this file.
"""
from __future__ import annotations
from typing import List

from app.llm_client import call_json
from app.schemas import ClaimElement, EvidenceMatch, VerificationResult

SYSTEM_PROMPT = """You are a strict patent claim verifier. Given a claim \
element and the best-matching passage retrieved from the inventor's \
disclosure, decide if the claim element is fully supported by the \
disclosure. Mark UNSUPPORTED if the claim element introduces any \
technical detail, mechanism, or specificity not actually present in the \
disclosure passage — even if it sounds plausible or is a reasonable \
engineering choice. Being plausible is not the same as being disclosed."""

JSON_SCHEMA_INSTRUCTIONS = """Return JSON with this exact shape:
{"status": "SUPPORTED" or "UNSUPPORTED", "reasoning": "one concise sentence"}"""


def verify_elements(
    claim_elements: List[ClaimElement], evidence_matches: List[EvidenceMatch]
) -> List[VerificationResult]:
    evidence_by_id = {e.element_id: e for e in evidence_matches}
    results: List[VerificationResult] = []

    for el in claim_elements:
        evidence = evidence_by_id.get(el.element_id)
        passage = evidence.passage if evidence else ""
        hybrid_score = evidence.hybrid_score if evidence else 0.0

        user_prompt = (
            f"{JSON_SCHEMA_INSTRUCTIONS}\n\n"
            f"Claim element:\n\"{el.text}\"\n\n"
            f"Best-matching disclosure passage (retrieval score {hybrid_score:.2f}, "
            f"low score may mean no real match exists):\n\"{passage or '(no passage found)'}\""
        )

        try:
            parsed = call_json(SYSTEM_PROMPT, user_prompt)
            status = parsed.get("status", "UNSUPPORTED")
            reasoning = parsed.get("reasoning", "")
        except ValueError:
            status, reasoning = "UNSUPPORTED", "Verification call failed; treated as unverified."

        results.append(
            VerificationResult(
                element_id=el.element_id,
                element_text=el.text,
                status=status,
                reasoning=reasoning,
                evidence_passage=passage or None,
                hybrid_score=hybrid_score,
            )
        )
    return results
