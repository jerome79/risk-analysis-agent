import argparse

from risk_analysis_agent.ingest import ingest_folder
from risk_analysis_agent.retriever import index_dataframe

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/samples")
    args = ap.parse_args()

    df = ingest_folder(args.folder)
    index_dataframe(df)
    print("Indexed:", len(df))
