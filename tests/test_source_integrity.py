"""Source-integrity checks that catch merge debris before import-time failures."""

import compileall
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_source_and_tests_compile_cleanly():
    """Compile all source and test files to catch syntax-level merge debris."""
    assert compileall.compile_dir(str(REPO_ROOT / "src"), quiet=1)
    assert compileall.compile_dir(str(REPO_ROOT / "tests"), quiet=1)
