"""Exercise the coverage summary script with real coverage data."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from coverage import Coverage

SUMMARY_SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/coverage_summary.py"


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_summary_reports_missing_lines_and_marks_incomplete_runs(tmp_path, outcome):
    source = tmp_path / "example.py"
    source.write_text("value = 1\nif value == 2:\n    value = 3\n")
    data_file = tmp_path / ".coverage"
    coverage = Coverage(data_file=str(data_file), branch=True, config_file=False)
    coverage.start()
    runpy.run_path(str(source))
    coverage.stop()
    coverage.save()

    result = subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT), "--test-outcome", outcome],
        cwd=tmp_path,
        env={**os.environ, "COVERAGE_FILE": str(data_file)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert "## Test coverage:" in result.stdout
    assert "example.py" in result.stdout
    assert "Missing" in result.stdout
    assert "3" in result.stdout
    assert ("**Partial report:**" in result.stdout) == (outcome != "success")


def test_summary_without_data_does_not_claim_zero_coverage(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT), "--test-outcome", "failure"],
        cwd=tmp_path,
        env={**os.environ, "COVERAGE_FILE": str(tmp_path / "missing.coverage")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", "skipped"])
def test_required_check_passes_only_when_shared_verification_succeeds(outcome):
    workflow_path = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yaml"
    workflow = yaml.safe_load(workflow_path.read_text())
    gate = workflow["jobs"]["lint-and-test"]
    assert gate["name"] == "lint and test"
    assert gate["needs"] == "verify"
    assert gate["if"] == "${{ always() }}"
    step = gate["steps"][0]
    assert step["env"]["VERIFY_RESULT"] == "${{ needs.verify.result }}"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={**os.environ, "VERIFY_RESULT": outcome},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (outcome == "success")
