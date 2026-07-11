"""
Shared data contracts for the Patent Verification & Evaluation Agent System.
Every agent reads from and writes to these Pydantic models, which also
constitute the LangGraph state.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ClaimElement(BaseModel):
    """A single atomic technical element extracted from a patent claim."""
    element_id: str
    text: str
    claim_number: Optional[int] = None  # which claim (1, 2, 3...) this element belongs to


class EvidenceMatch(BaseModel):
    """Result of hybrid retrieval for one claim element against the disclosure."""
    element_id: str
    passage: str
    bm25_score: float
    vector_score: float
    hybrid_score: float


class VerificationResult(BaseModel):
    """Verdict for a single claim element."""
    element_id: str
    element_text: str
    status: str  # "SUPPORTED" | "UNSUPPORTED"
    reasoning: str
    evidence_passage: Optional[str] = None
    hybrid_score: float = 0.0


class CoverageGap(BaseModel):
    """An invention concept present in the disclosure but absent from the claims."""
    concept: str
    disclosure_evidence: Optional[str] = None


class QualityReport(BaseModel):
    """Final aggregated output of the whole pipeline."""
    coverage_score: float  # 0-100, % of claim elements that are SUPPORTED
    hallucination_count: int
    supported_count: int
    unsupported_count: int
    total_elements: int
    missing_concepts: List[CoverageGap] = Field(default_factory=list)
    verifications: List[VerificationResult] = Field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Coverage score: {self.coverage_score:.1f}%",
            f"Supported elements: {self.supported_count}/{self.total_elements}",
            f"Hallucinations flagged: {self.hallucination_count}",
            f"Missing concepts: {len(self.missing_concepts)}",
        ]
        return "\n".join(lines)


class PipelineState(BaseModel):
    """The full state object threaded through the LangGraph graph."""
    disclosure: str
    draft_claims_text: str

    claim_elements: List[ClaimElement] = Field(default_factory=list)
    evidence_matches: List[EvidenceMatch] = Field(default_factory=list)
    verifications: List[VerificationResult] = Field(default_factory=list)
    missing_concepts: List[CoverageGap] = Field(default_factory=list)
    report: Optional[QualityReport] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
