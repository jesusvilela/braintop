"""The solution-plan example must keep running and keep its documented numbers."""

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "solution_plan_implementation.py"


def test_solution_plan_runs_and_matches_documented_figures(tmp_path, monkeypatch, capsys):
    output = tmp_path / "plan.html"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "-o", str(output)])
    runpy.run_path(str(SCRIPT), run_name="__main__")

    out = capsys.readouterr().out
    assert output.stat().st_size > 0
    assert "FAILED" not in out
    # Figures quoted in SOLUTION_PLAN.md
    assert "nodes: 30, edges: 86" in out
    assert "'llm_similarity': 4" in out
    assert "embedding dimension: 768" in out
    assert "variance kept by 3 principal components: 43.2%" in out
