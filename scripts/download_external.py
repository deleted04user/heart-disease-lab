"""Retrieve the separate official UCI Hungarian cohort without overwriting files."""

import argparse
import json
from pathlib import Path

from _common import configure_logging
from heart_disease.external import EXTERNAL_PATH, download_external

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXTERNAL_PATH)
    args = parser.parse_args()
    configure_logging()
    try:
        dataset = download_external(args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"External acquisition failed: {exc}\n")
    print(json.dumps(dataset.metadata, indent=2))
