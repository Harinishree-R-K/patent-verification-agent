"""
Dry-run test: mocks app.llm_client.call_json so the full LangGraph pipeline
can be validated end-to-end WITHOUT a real API key. This proves the graph
wiring, state passing, and aggregation logic are correct. It does not prove
the LLM prompts produce good judgments — that requires a real key and
real testing, which you should do before pitching this anywhere.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch

from app.graph import run_pipeline

SAMPLE_DISCLOSURE = (
    "The invention is a smart water bottle. It has an internal weight-based "
    "sensor that measures liquid consumed. A microcontroller logs each sip "
    "with a timestamp. The bottle connects to a mobile app over a wired USB "
    "sync cable."
)

SAMPLE_CLAIMS = (
    "1. A device comprising a weight-based sensor to measure liquid consumed; "
    "a microcontroller to log sip events with timestamps; and a wireless "
    "Bluetooth transceiver to stream data to a mobile app."
)


def fake_call_json(system: str, user: str, max_tokens: int = 1000):
    """Returns canned responses depending on which agent is calling."""
    if "break each claim into its atomic technical elements" in system:
        return {
            "elements": [
                {"claim_number": 1, "text": "a weight-based sensor to measure liquid consumed"},
                {"claim_number": 1, "text": "a microcontroller to log sip events with timestamps"},
                {"claim_number": 1, "text": "a wireless Bluetooth transceiver to stream data to a mobile app"},
            ]
        }
    if "strict patent claim verifier" in system:
        if "Bluetooth" in user:
            return {"status": "UNSUPPORTED", "reasoning": "Disclosure specifies a wired USB cable, not wireless Bluetooth."}
        return {"status": "SUPPORTED", "reasoning": "Directly matches the disclosure passage."}
    if "patent coverage analyst" in system:
        return {"missing_concepts": []}
    raise AssertionError(f"Unexpected system prompt in test: {system[:80]}")


def test_pipeline_end_to_end_with_mocked_llm():
    with patch("app.agents.claim_extraction.call_json", side_effect=fake_call_json), \
         patch("app.agents.verification.call_json", side_effect=fake_call_json), \
         patch("app.agents.coverage_analysis.call_json", side_effect=fake_call_json):

        final_state = run_pipeline(SAMPLE_DISCLOSURE, SAMPLE_CLAIMS)

    report = final_state.report
    assert report is not None, "Pipeline did not produce a report"
    assert report.total_elements == 3
    assert report.supported_count == 2
    assert report.unsupported_count == 1
    assert report.hallucination_count == 1
    assert 60 < report.coverage_score < 70

    statuses = {v.element_text[:20]: v.status for v in report.verifications}
    bluetooth_result = [v for v in report.verifications if "Bluetooth" in v.element_text][0]
    assert bluetooth_result.status == "UNSUPPORTED"

    print("\n--- Quality Report ---")
    print(report.summary())
    print("----------------------")


if __name__ == "__main__":
    test_pipeline_end_to_end_with_mocked_llm()
    print("\nDry-run pipeline test PASSED.")
