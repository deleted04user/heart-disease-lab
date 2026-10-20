"""Offline scanner-result regressions; the live advisory probe belongs to CI."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_gate", Path(__file__).resolve().parents[1] / "scripts/verify_audit_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture_report(**changes):
    return {
        "dependencies": [
            {
                "name": "urllib3",
                "version": "1.26.4",
                "vulns": [{"id": "PYSEC-2021-108", "aliases": ["CVE-2021-33503"]}],
                **changes,
            }
        ]
    }


def test_expected_advisory_is_required_in_primary_id_or_alias():
    assert gate.validate_detection(1, fixture_report()) == ["CVE-2021-33503", "PYSEC-2021-108"]
    assert gate.validate_detection(
        1,
        fixture_report(
            vulns=[
                {
                    "id": "provider-specific",
                    "aliases": ["GHSA-q2q7-5pp4-w6pg"],
                }
            ]
        ),
    ) == ["GHSA-q2q7-5pp4-w6pg"]


@pytest.mark.parametrize(
    "code,report",
    [
        (0, fixture_report()),
        (2, fixture_report()),
        (1, {}),
        (1, {"dependencies": []}),
        (1, fixture_report(name="different-package")),
        (1, fixture_report(version="2.0.0")),
        (1, fixture_report(skip_reason="advisory service unavailable")),
        (1, fixture_report(vulns=[])),
        (1, fixture_report(vulns=[{"id": "unrelated-advisory"}])),
        (1, fixture_report(vulns=[{"id": ["invalid identifier"]}])),
        (1, fixture_report(vulns=[{"id": "PYSEC-2021-108", "aliases": "invalid"}])),
    ],
)
def test_clean_partial_wrong_and_error_results_cannot_pass(code, report):
    with pytest.raises(ValueError):
        gate.validate_detection(code, report)


def test_live_command_scans_a_temporary_pin_without_using_pip_or_installing(monkeypatch):
    observed = []

    def audit(command, **kwargs):
        path = Path(command[command.index("-r") + 1])
        assert path.read_text() == "urllib3==1.26.4\n"
        assert "--disable-pip" in command and "--no-deps" in command
        assert "--strict" in command and "--ignore-vuln" not in command
        assert "install" not in command and "--fix" not in command
        assert kwargs["timeout"] == 90
        observed.append(path)
        return subprocess.CompletedProcess(command, 1, json.dumps(fixture_report()), "")

    monkeypatch.setattr(gate.subprocess, "run", audit)
    result = gate.verify_gate()
    assert result["fixture_installed"] is False and result["status"] == "passed"
    assert not observed[0].exists()


def test_non_json_scanner_failure_is_not_success(monkeypatch):
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [],
            1,
            "",
            "network failed",
        ),
    )
    with pytest.raises(ValueError, match="no valid JSON"):
        gate.verify_gate()
