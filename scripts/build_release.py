"""Build the portable, verified release and pin the cloud deployment checksum."""

import argparse
import json
from pathlib import Path

from _common import ROOT
from heart_disease import __version__
from heart_disease.bundle import build_bundle
from heart_disease.config import MODEL_PATH, REPORT_DIR
from heart_disease.io import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=__version__)
    parser.add_argument("--repository", default="deleted04user/heart-disease-lab")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.output = args.output or ROOT / f"dist/heart-disease-lab-{args.version}.zip"
    result = build_bundle(ROOT, MODEL_PATH, REPORT_DIR, args.output, version=args.version)
    url = (
        f"https://github.com/{args.repository}/releases/download/v{args.version}/{args.output.name}"
    )
    write_json(
        ROOT / "deploy/release.json",
        {key: result[key] for key in ["version", "asset", "sha256", "run_id"]}
        | {"repository": args.repository, "url": url},
    )
    (ROOT / "deploy/requirements.txt").write_text(result["runtime_requirements"], encoding="utf-8")
    write_json(ROOT / "dist/release_summary.json", result)
    print(json.dumps(result, indent=2))
