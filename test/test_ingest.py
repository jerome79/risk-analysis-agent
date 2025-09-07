import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path

import pandas as pd

from app.ingest import SCHEMA, ingest_folder


def test_ingest_creates_expected_schema_and_rows(tmp_path: Path) -> None:
    """
    Test that ingest_folder creates a DataFrame with the expected schema and rows
    for a sample folder structure and file.
    """
    issuer = "ACME_CORP"
    year = "2024"
    d = tmp_path / issuer / year
    d.mkdir(parents=True, exist_ok=True)
    sample = d / "item_1a.txt"
    sample.write_text(
        "This section discusses key market and liquidity risks affecting operations.",
        encoding="utf-8",
    )

    df = ingest_folder(str(tmp_path))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == SCHEMA
    # sanity checks
    assert (df["issuer"] == issuer).all()
    assert (df["fiscal_year"] == year).all()
    assert (df["section"] == "Item 1A").any()
    assert df["chunk_id"].str.contains("item_1a.txt").any()
