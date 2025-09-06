import argparse
from app.ingest import ingest_folder
from app.retriever import index_dataframe

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/samples")
    ap.add_argument("--persist", default=None)
    args = ap.parse_args()

    df = ingest_folder(args.folder)
    index_dataframe(df, args.persist)
    print("Indexed:", len(df))
