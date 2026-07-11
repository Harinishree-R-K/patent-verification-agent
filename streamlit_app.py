"""
Streamlit UI for the Patent Verification & Evaluation Agent System.

Run with:
    streamlit run streamlit_app.py

This calls the LangGraph pipeline in-process (no need to run the FastAPI
server separately). The FastAPI app in app/main.py exposes the same
pipeline as an API, for integrating into a real product.
"""
import streamlit as st

from app.graph import run_pipeline

st.set_page_config(page_title="Patent Verification Agent", layout="wide")

st.title("Patent Verification & Evaluation Agent System")
st.caption(
    "Verifies AI-generated patent claims against inventor disclosures. "
    "Grounds every claim element, flags hallucinations, surfaces coverage gaps."
)

with open("sample_data/sample_disclosure.txt") as f:
    default_disclosure = f.read()
with open("sample_data/sample_draft_claims.txt") as f:
    default_claims = f.read()

col1, col2 = st.columns(2)
with col1:
    disclosure = st.text_area("Inventor Disclosure", value=default_disclosure, height=280)
with col2:
    draft_claims_text = st.text_area("AI-Drafted Patent Claims", value=default_claims, height=280)

if st.button("Run Verification", type="primary"):
    with st.spinner("Running claim extraction, retrieval, verification, coverage analysis..."):
        try:
            final_state = run_pipeline(disclosure, draft_claims_text)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    report = final_state.report
    if report is None:
        st.error("No report produced.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coverage Score", f"{report.coverage_score:.1f}%")
    c2.metric("Hallucinations Flagged", report.hallucination_count)
    c3.metric("Supported Elements", f"{report.supported_count}/{report.total_elements}")
    c4.metric("Missing Concepts", len(report.missing_concepts))

    st.subheader("Claim-by-claim verdicts")
    for v in report.verifications:
        icon = "✅" if v.status == "SUPPORTED" else "❌"
        with st.expander(f"{icon} {v.status} — {v.element_text[:80]}"):
            st.write(f"**Reasoning:** {v.reasoning}")
            if v.evidence_passage:
                st.write(f"**Nearest disclosure evidence** (hybrid score {v.hybrid_score:.2f}):")
                st.code(v.evidence_passage, language=None)

    if report.missing_concepts:
        st.subheader("Coverage gaps — in disclosure, missing from claims")
        for gap in report.missing_concepts:
            st.write(f"- **{gap.concept}**" + (f" — _{gap.disclosure_evidence}_" if gap.disclosure_evidence else ""))
