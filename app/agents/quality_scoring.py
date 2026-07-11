"""
Agent 5: Quality Scoring Agent

Pure aggregation — no LLM call. Takes the outputs of verification and
coverage analysis and rolls them into the final QualityReport. Kept as
plain arithmetic deliberately: scoring logic should be transparent and
auditable, not another opaque model call.
"""
from __future__ import annotations
from typing import List

from app.schemas import CoverageGap, QualityReport, VerificationResult


def score_quality(
    verifications: List[VerificationResult],
    missing_concepts: List[CoverageGap],
) -> QualityReport:
    total = len(verifications)
    supported = sum(1 for v in verifications if v.status == "SUPPORTED")
    unsupported = total - supported
    coverage_score = (supported / total * 100) if total > 0 else 0.0

    return QualityReport(
        coverage_score=round(coverage_score, 1),
        hallucination_count=unsupported,
        supported_count=supported,
        unsupported_count=unsupported,
        total_elements=total,
        missing_concepts=missing_concepts,
        verifications=verifications,
    )
