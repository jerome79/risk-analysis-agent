# scripts/cli.py
import argparse
import os
import subprocess
import sys


def serve() -> int:
    """
    Launches the Streamlit dashboard for the risk analysis agent.

    Returns:
        int: The exit code from the Streamlit process.
    """
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "risk_analysis_agent/ui_streamlit.py"])


def benchmark(args: argparse.Namespace) -> int:
    """
    Runs the benchmark script with the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        int: The exit code from the benchmark process.
    """
    cmd = [sys.executable, "scripts/benchmark.py", "--csv", args.csv, "--model", args.model]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.results:
        cmd += ["--results", args.results]
    return subprocess.call(cmd)


def demo() -> int:
    """
    Creates a tiny sample CSV file if it does not exist and launches the Streamlit dashboard.

    Returns:
        int: The exit code from the Streamlit process.
    """
    os.makedirs("data", exist_ok=True)
    sample = "data/sample_tiny.csv"
    if not os.path.exists(sample):
        with open(sample, "w") as f:
            f.write(
                "date,ticker,headline,text\n2025-01-01,AAPL,Apple rises,Apple stock surges after earnings\n2025-01-01,TSLA,Tesla falls,Tesla shares dip after recall news\n"
            )
    return serve()


def main() -> None:
    """
    Entry point for the CLI. Parses arguments and dispatches to the appropriate command.
    """
    p = argparse.ArgumentParser(prog="msa")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("serve", help="Run Streamlit dashboard")
    s1.set_defaults(func=lambda _: serve())

    s2 = sub.add_parser("benchmark", help="Run throughput benchmark")
    s2.add_argument("--csv", required=True)
    s2.add_argument("--model", required=True)
    s2.add_argument("--batch-size", type=int)
    s2.add_argument("--limit", type=int)
    s2.add_argument("--results")
    s2.set_defaults(func=benchmark)

    s3 = sub.add_parser("demo", help="Create tiny sample & run dashboard")
    s3.set_defaults(func=lambda _: demo())

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
