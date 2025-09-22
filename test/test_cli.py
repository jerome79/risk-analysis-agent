# test/test_cli_smoke.py
import subprocess
import sys


def test_cli_help() -> None:
    """Test that the CLI help command runs successfully and exits with code 0."""
    assert subprocess.call([sys.executable, "scripts/cli.py", "-h"]) == 0
