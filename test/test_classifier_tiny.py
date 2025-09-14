import sys
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


from risk_analysis_agent.classifier import ZeroShotRisk


def test_zero_shot_classifier_with_tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test ZeroShotRisk classifier using a tiny MNLI model for fast CI.
    Sets the ZSL_MODEL environment variable, classifies sample texts,
    and asserts output structure and probability bounds.
    """
    monkeypatch.setenv("ZSL_MODEL", "sshleifer/tiny-distilbart-mnli")
    zsl = ZeroShotRisk()

    texts = [
        "Interest rate volatility could affect our funding costs.",
        "There is a risk of cyber attacks on our infrastructure.",
    ]
    out = zsl.classify(texts, top_k=2)
    baseline_low = 1
    baseline_up = 2
    assert len(out) == len(texts)
    for labels in out:
        assert baseline_low <= len(labels) <= baseline_up
        for lab, prob in labels:
            assert isinstance(lab, str)
            assert 0.0 <= prob <= 1.0
