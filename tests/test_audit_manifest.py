"""Validate the cap-bound evidence ledger's structure and referents.

This is a manifest integrity check, NOT verification of the mathematical
theorem or a substitute for an independent external reviewer.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "audits" / "nonconvex_cap_upper_2026-11-16.json"
REQUIRED = {
    "task_id", "claim", "specification", "implementation", "metric",
    "tests", "controls", "preregistration", "evidence", "round_trip",
    "result",
}


def test_audit_manifest_contains_provenance_and_nonoptimality_boundary():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(data) == REQUIRED
    assert data["claim"]["status_before"] == "H"
    assert data["result"]["status_after"] == "A"
    assert data["metric"]["type"] == "typed_tuple"
    assert data["metric"]["direction"] == "two_sided"
    assert data["metric"]["raw_before_clipping"] is True
    assert data["metric"]["semantic_invariance_claimed"] is True
    assert "optimal" in " ".join(data["claim"]["forbidden_promotions"]).lower()
    assert data["preregistration"]["kill_condition"]
    assert data["preregistration"]["stop_condition"]
    assert data["result"]["active_remainder"]
    assert data["evidence"]["execution_records"]
    assert data["round_trip"]["edges"]
    assert any(
        "not an independent verification" in item["origin"]
        for item in data["evidence"]["independent_witnesses"]
    )

    for path in [
        "src/braintop/core/correspondence.py",
        "tests/test_global_upper_envelope.py",
        "examples/nonconvex_cap_lower_sweep.py",
        "docs/CORRESPONDENCE.md",
    ]:
        assert (ROOT / path).is_file(), path
