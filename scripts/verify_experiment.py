"""Verify a completed experiment offline without loading models or source records."""

import argparse
import json
from pathlib import Path

from _common import configure_logging
from heart_disease.experiments import verify_run_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Directory containing run.json")
    args = parser.parse_args()
    configure_logging()
    try:
        manifest = verify_run_snapshot(args.run_directory)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(1, f"Experiment verification failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "run_id": manifest["run_id"],
                "selected_model": manifest["selected_model"],
                "model_sha256": manifest["model_sha256"],
                "report_count": len(manifest["reports_sha256"]),
                "metadata_checksum_recorded": "metadata_sha256" in manifest,
                "model_deserialized": False,
                "source_records_read": False,
                "scope": "Snapshot consistency only; not authenticity, metric recomputation or clinical validation.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
