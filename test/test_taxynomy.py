import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.taxonomy import canonical_labels, to_key


def test_taxonomy_labels_and_keys() -> None:
    """
    Test that canonical_labels returns a list of labels and to_key produces valid keys.
    """
    labels = canonical_labels()
    baseline = 5
    assert isinstance(labels, list)
    assert len(labels) >= baseline  # baseline sanity
    # keys should be lowercase, spaces->underscores, slashes removed
    for lab in labels:
        k = to_key(lab)
        assert k == k.lower()
        assert "/" not in k
        assert " " not in k
