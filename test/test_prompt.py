import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from risk_analysis_agent.prompts import QA_PROMPT, RISK_SUMMARY_PROMPT


def test_risk_summary_prompt_exists() -> None:
    """
    Test that RISK_SUMMARY_PROMPT exists and contains required placeholders.
    """
    assert isinstance(RISK_SUMMARY_PROMPT, str)
    assert "{issuer}" in RISK_SUMMARY_PROMPT
    assert "{year}" in RISK_SUMMARY_PROMPT
    assert "{context}" in RISK_SUMMARY_PROMPT


def test_qa_prompt_exists() -> None:
    """
    Test that QA_PROMPT exists and contains required placeholders.
    """
    assert isinstance(QA_PROMPT, str)
    assert "{question}" in QA_PROMPT
    assert "{context}" in QA_PROMPT
