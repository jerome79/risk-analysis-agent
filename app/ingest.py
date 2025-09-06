import os
from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

SCHEMA = ["issuer","fiscal_year","section","filepath","text","chunk_id"]

def _resolve_dir(path: str | None) -> Path:
    root = Path(__file__).resolve().parents[1]
    p = Path(path or "data/samples")
    return (p if p.is_absolute() else (root / p)).resolve()

def _read_txt(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def ingest_folder(folder: str | None) -> pd.DataFrame:
    base = _resolve_dir(folder)
    fps = list(base.rglob("*.txt"))
    rows = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    for fp in fps:
        parts = fp.parts
        issuer = parts[-3] if len(parts) >= 3 else "UNKNOWN_ISSUER"
        fiscal_year = parts[-2] if len(parts) >= 2 else "UNKNOWN_YEAR"
        name = fp.name.lower()
        section = "Item 1A" if ("1a" in name or "risk" in name) else "unknown"
        chunks = splitter.split_text(_read_txt(fp))
        for i, ch in enumerate(chunks):
            rows.append({
                "issuer": issuer,
                "fiscal_year": fiscal_year,
                "section": section,
                "filepath": str(fp),
                "text": ch,
                "chunk_id": f"{fp.name}:::{i}"
            })
    df = pd.DataFrame(rows, columns=SCHEMA)
    return df

def save_parquet(df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
