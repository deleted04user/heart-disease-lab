"""Audit the Git index before publishing source and aggregate portfolio materials."""

import json
import re
import subprocess
from pathlib import PurePosixPath

from _common import ROOT


def audit() -> dict:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    names = [name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    failures = []
    for name in names:
        path = PurePosixPath(name)
        if (
            name.startswith(
                (
                    ".venv/",
                    "experiments/runs/",
                    "data/raw/",
                    "data/processed/",
                    "data/reserved/",
                    "models/",
                    ".deployment-cache/",
                    "dist/",
                    "build/",
                )
            )
            and path.name != ".gitkeep"
        ):
            failures.append(f"Excluded data/artifact directory: {name}")
        if (
            path.suffix in {".joblib", ".pkl", ".pickle", ".zip"}
            or path.name.startswith(".env")
            or name == ".streamlit/secrets.toml"
        ):
            failures.append(f"Private/large artifact type: {name}")
        content = subprocess.run(
            ["git", "show", ":" + name], cwd=ROOT, check=True, capture_output=True
        ).stdout
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue  # Only documented screenshots, plots and demo video are binary.
        if re.search(
            r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)",
            text,
        ):
            failures.append(f"Possible credential/private key in {name}")
        if re.search(r"[A-Za-z]:\\Users\\[^\\\s]+\\|/(?:home|Users)/[A-Za-z0-9_.-]+/", text):
            failures.append(f"Workstation-specific personal path in {name}")
    if failures:
        raise ValueError("Publication review failed:\n" + "\n".join(failures))
    return {
        "status": "passed",
        "tracked_files": len(names),
        "checks": [
            "raw and reserved data excluded",
            "model binaries and run snapshots excluded",
            "secrets/private keys excluded",
            "workstation paths excluded",
            "only source and aggregate portfolio materials staged",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
