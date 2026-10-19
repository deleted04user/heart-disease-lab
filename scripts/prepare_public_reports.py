"""Replace workstation paths in active aggregate reports with project-relative paths.

The immutable local experiment snapshots and ignored model metadata are retained
unchanged. Public report values and experimental metrics are not recomputed.
"""
import json
from pathlib import Path

from _common import ROOT
from heart_disease.config import REPORT_DIR


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + "\\", "").replace(ROOT.as_posix() + "/", "").replace(str(ROOT), ".")
    return value


if __name__ == "__main__":
    changed = []
    for path in REPORT_DIR.rglob("*"):
        if not path.is_file() or path.name == "active_run.json" or path.suffix not in {".json", ".md", ".txt", ".csv"}:
            continue
        original = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            updated = json.dumps(clean(json.loads(original)), indent=2, allow_nan=False) + "\n"
        else:
            updated = clean(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.relative_to(ROOT).as_posix())
    print(json.dumps({"sanitized_files": changed, "metrics_changed": False, "immutable_snapshots_modified": False}, indent=2))
