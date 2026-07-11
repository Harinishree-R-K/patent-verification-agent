"""
LangGraph orchestration for the Patent Verification & Evaluation
Agent System.

Pipeline:
  claim_extraction -> evidence_retrieval -> verification
                                          -> coverage_analysis
                    -> quality_scoring -> END

verification and coverage_analysis both depend on claim_extraction's
output but not on each other, so they run as parallel branches that
join at quality_scoring.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agents.claim_extraction import extract_claim_elements
from app.agents.coverage_analysis import analyze_coverage
from app.agents.evidence_retrieval import retrieve_evidence
from app.agents.quality_scoring import score_quality
from app.agents.verification import verify_elements
from app.schemas import PipelineState


def _node_claim_extraction(state: PipelineState) -> dict:
    elements = extract_claim_elements(state.draft_claims_text)
    return {"claim_elements": elements}


def _node_evidence_retrieval(state: PipelineState) -> dict:
    matches = retrieve_evidence(state.disclosure, state.claim_elements)
    return {"evidence_matches": matches}


def _node_verification(state: PipelineState) -> dict:
    results = verify_elements(state.claim_elements, state.evidence_matches)
    return {"verifications": results}


def _node_coverage_analysis(state: PipelineState) -> dict:
    gaps = analyze_coverage(state.disclosure, state.claim_elements)
    return {"missing_concepts": gaps}


def _node_quality_scoring(state: PipelineState) -> dict:
    report = score_quality(state.verifications, state.missing_concepts)
    return {"report": report}


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("claim_extraction", _node_claim_extraction)
    graph.add_node("evidence_retrieval", _node_evidence_retrieval)
    graph.add_node("verification", _node_verification)
    graph.add_node("coverage_analysis", _node_coverage_analysis)
    graph.add_node("quality_scoring", _node_quality_scoring)

    graph.set_entry_point("claim_extraction")
    graph.add_edge("claim_extraction", "evidence_retrieval")
    graph.add_edge("evidence_retrieval", "verification")
    graph.add_edge("claim_extraction", "coverage_analysis")
    graph.add_edge("verification", "quality_scoring")
    graph.add_edge("coverage_analysis", "quality_scoring")
    graph.add_edge("quality_scoring", END)

    return graph.compile()


def run_pipeline(disclosure: str, draft_claims_text: str) -> PipelineState:
    """Convenience entrypoint used by both the FastAPI route and Streamlit UI."""
    app_graph = build_graph()
    initial_state = PipelineState(
        disclosure=disclosure, draft_claims_text=draft_claims_text
    )
    final_state_dict = app_graph.invoke(initial_state)
    return PipelineState(**final_state_dict)
