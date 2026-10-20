"""Verify pip-audit rejects a known vulnerable pin without installing that package."""

import argparse
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

PACKAGE, VERSION = "urllib3", "1.26.4"
EXPECTED_IDS = {"PYSEC-2021-108", "CVE-2021-33503", "GHSA-q2q7-5pp4-w6pg"}
# https://github.com/urllib3/urllib3/security/advisories/GHSA-q2q7-5pp4-w6pg


def validate_detection(returncode: int, report: dict) -> list[str]:
    """An error exit alone is insufficient: require the expected advisory evidence."""
    if returncode != 1:
        raise ValueError("The vulnerable fixture must produce pip-audit exit code 1.")
    dependencies = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(dependencies, list) or len(dependencies) != 1:
        raise ValueError("The scanner must report exactly the single fixture dependency.")
    dependency = dependencies[0]
    if (
        not isinstance(dependency, dict)
        or dependency.get("name") != PACKAGE
        or dependency.get("version") != VERSION
        or "skip_reason" in dependency
    ):
        raise ValueError("The fixture dependency was skipped or differs from the requested pin.")
    findings = dependency.get("vulns")
    if not isinstance(findings, list):
        raise ValueError("Missing vulnerability findings in the scanner report.")
    matched = set()
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("Malformed vulnerability finding in the scanner report.")
        aliases = finding.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise ValueError("Malformed advisory aliases in the scanner report.")
        identifier = finding.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Malformed advisory identifier in the scanner report.")
        identifiers = [identifier, *aliases]
        matched.update(identifier for identifier in identifiers if identifier in EXPECTED_IDS)
    if not matched:
        raise ValueError("The expected historical vulnerability was not detected.")
    return sorted(matched)


def verify_gate() -> dict:
    with tempfile.TemporaryDirectory(prefix="heart-audit-gate-") as directory:
        requirement = Path(directory) / "fixture.txt"
        requirement.write_text(f"{PACKAGE}=={VERSION}\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "-r",
                str(requirement),
                "--no-deps",
                "--disable-pip",
                "--strict",
                "--progress-spinner",
                "off",
                "--vulnerability-service",
                "pypi",
                "--format",
                "json",
                "--aliases",
                "on",
                "--output",
                "-",
                "--timeout",
                "15",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
            check=False,
        )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The scanner returned no valid JSON; an audit error is not detection."
        ) from exc
    matched = validate_detection(result.returncode, report)
    return {
        "status": "passed",
        "scanner": "pip-audit",
        "scanner_version": version("pip-audit"),
        "auditor_exit_code": result.returncode,
        "fixture": {"package": PACKAGE, "version": VERSION},
        "matched_advisory_ids": matched,
        "fixture_installed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        report = verify_gate()
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Dependency audit gate verification failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
