"""Acquire the official original Cleveland subset without overwriting data."""

import argparse
import json
from pathlib import Path

from _common import configure_logging
from heart_disease.config import DATA_PATH
from heart_disease.data import download_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    configure_logging()
    try:
        dataset = download_dataset(args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Data acquisition failed: {exc}\n")
    print(json.dumps(dataset.metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
