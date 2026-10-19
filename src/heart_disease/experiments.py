"""Preserve immutable run folders and publish only a fully completed experiment."""

import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .config import FEATURES, ROOT
from .io import read_json, sha256, write_json
from .prediction import metadata_path

RUNS_DIR = ROOT / "experiments/runs"


def _report_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or relative.is_absolute()
        or relative.as_posix() != name
        or any(part in {".", ".."} for part in name.split("/"))
    ):
        raise ValueError("Unsafe report path in experiment manifest.")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Report path escapes the experiment directory.")
    return path


def _report_inventory(run_dir: Path) -> dict[str, Path]:
    root = run_dir / "reports"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Experiment reports must be a local directory, not a symbolic link.")
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symbolic links are not accepted in experiment reports.")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            result[name] = _report_path(root, name)
    return result


def _identity(report: dict) -> dict:
    return {
        "run_id": report.get("run_id"),
        "selected_model": report.get("selected_model"),
        "probability_method": report.get("calibration", {}).get("selected_method", "raw"),
        "data_sha256": report.get("data", {}).get("sha256"),
    }


def _verify_metadata(metadata: dict, identity: dict, model: Path) -> None:
    expected = {
        "run_id": metadata.get("run_id"),
        "selected_model": metadata.get("selected_model"),
        "probability_method": metadata.get("probability_method", "raw"),
        "data_sha256": metadata.get("data", {}).get("sha256"),
    }
    if (
        not all(isinstance(value, str) and value for value in identity.values())
        or expected != identity
        or metadata.get("artifact_schema") != 1
        or metadata.get("features") != FEATURES
        or metadata.get("model_sha256") != sha256(model)
    ):
        raise ValueError("Model, metadata and report integrity or run identity differs.")


def verify_run_snapshot(run_dir: Path) -> dict:
    """Check a completed snapshot without loading pickle objects or source data.

    Legacy manifests have no metadata digest; their identity is still checked.
    Hashes detect inconsistency, not authenticity of an untrusted snapshot.
    """
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run.json"
    if manifest_path.is_symlink() or (run_dir / "models").is_symlink():
        raise ValueError("Symbolic links are not accepted in experiment artifacts.")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("schema", 1) not in {1, 2}:
        raise ValueError("Only a supported completed run can be verified or published.")
    model = run_dir / "models/heart_model.joblib"
    metadata_file = metadata_path(model)
    if model.is_symlink() or metadata_file.is_symlink():
        raise ValueError("Symbolic links are not accepted in experiment artifacts.")
    if sha256(model) != manifest.get("model_sha256"):
        raise ValueError("Completed model integrity check failed; publishing refused.")
    if manifest.get("schema") == 2 and sha256(metadata_file) != manifest.get("metadata_sha256"):
        raise ValueError("Completed metadata integrity check failed; publishing refused.")
    files = manifest.get("reports_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("A nonempty report checksum inventory is required.")
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("Invalid report checksum inventory.")
        _report_path(run_dir / "reports", name)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid report SHA-256 digest.")
    inventory = _report_inventory(run_dir)
    if set(inventory) != set(files):
        raise ValueError("Report inventory differs from the completed manifest.")
    for name, path in inventory.items():
        if sha256(path) != files[name]:
            raise ValueError("Completed report integrity check failed; publishing refused.")
    if manifest.get("report") != "reports/evaluation_report.json":
        raise ValueError("A canonical evaluation report is required.")
    report = read_json(run_dir / "reports/evaluation_report.json")
    identity = _identity(report)
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("Manifest and evaluation report describe different runs.")
    _verify_metadata(read_json(metadata_file), identity, model)
    return manifest


def create_run(runs_dir: Path = RUNS_DIR) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(runs_dir).resolve() / (stamp + "-" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    (path / "models").mkdir()
    (path / "reports").mkdir()
    return path


def preserve_existing(model_path: Path, report_dir: Path, runs_dir: Path = RUNS_DIR) -> Path | None:
    model_path, report_dir = Path(model_path), Path(report_dir)
    if not model_path.exists() or not metadata_path(model_path).exists():
        return None
    metadata = read_json(metadata_path(model_path))
    run_id = metadata["run_id"]
    if not run_id.isalnum():
        raise ValueError("Invalid existing run ID.")
    recorded_run = metadata.get("immutable_run_dir")
    if recorded_run:
        recorded = Path(recorded_run).resolve()
        if recorded.is_relative_to(Path(runs_dir).resolve()) and (recorded / "run.json").is_file():
            manifest = read_json(recorded / "run.json")
            if (
                manifest.get("status") == "complete"
                and manifest.get("run_id") == run_id
                and manifest.get("model_sha256") == sha256(model_path)
            ):
                return recorded
    destination = Path(runs_dir).resolve() / ("preserved-" + run_id)
    if destination.exists():
        if sha256(destination / "models/heart_model.joblib") != sha256(model_path):
            raise ValueError(
                "Preserved run exists with different model bytes; refusing to overwrite it."
            )
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "models").mkdir()
    shutil.copy2(model_path, destination / "models/heart_model.joblib")
    shutil.copy2(metadata_path(model_path), destination / "models/heart_model.metadata.json")
    if report_dir.exists():
        shutil.copytree(report_dir, destination / "reports")
    write_json(
        destination / "run.json",
        {
            "run_id": run_id,
            "status": "preserved",
            "source_model_sha256": sha256(model_path),
            "note": "Snapshot copied before publishing a newer experiment. Original report content retained.",
        },
    )
    return destination


def complete_run(run_dir: Path, report: dict):
    if (run_dir / "run.json").exists():
        raise ValueError("An experiment manifest already exists; refusing to rewrite the snapshot.")
    model = run_dir / "models/heart_model.joblib"
    metadata = read_json(metadata_path(model))
    _verify_metadata(metadata, _identity(report), model)
    if read_json(run_dir / "reports/evaluation_report.json") != report:
        raise ValueError("Saved evaluation report differs from the completed run.")
    hashes = {name: sha256(path) for name, path in _report_inventory(run_dir).items()}
    write_json(
        run_dir / "run.json",
        {
            "schema": 2,
            "run_id": report["run_id"],
            "status": "complete",
            "created_utc": report["created_utc"],
            "selected_model": report["selected_model"],
            "probability_method": report.get("calibration", {}).get("selected_method", "raw"),
            "data_sha256": report["data"]["sha256"],
            "model_sha256": sha256(model),
            "metadata_sha256": sha256(metadata_path(model)),
            "report": "reports/evaluation_report.json",
            "reports_sha256": hashes,
        },
    )


def _copy_atomic(source: Path, destination: Path):
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def publish_run(run_dir: Path, model_path: Path, report_dir: Path):
    manifest = verify_run_snapshot(run_dir)
    if any(
        Path(destination).resolve().is_relative_to(run_dir.resolve())
        for destination in (model_path, report_dir)
    ):
        raise ValueError("Active destinations cannot overwrite the immutable experiment snapshot.")
    source_model = run_dir / "models/heart_model.joblib"
    previous_manifest = Path(report_dir) / "active_run.json"
    if previous_manifest.exists():
        previous_files = set(read_json(previous_manifest).get("reports_sha256", {}))
        current_files = set(manifest.get("reports_sha256", {}))
        stale_paths = [
            _report_path(Path(report_dir), name) for name in previous_files - current_files
        ]
        for stale in stale_paths:
            if stale.is_file():
                stale.unlink()
    for name in manifest["reports_sha256"]:
        _copy_atomic(run_dir / "reports" / name, Path(report_dir) / name)
    _copy_atomic(source_model, Path(model_path))
    metadata = read_json(run_dir / "models/heart_model.metadata.json")
    metadata.update(
        {"report_dir": str(Path(report_dir).resolve()), "immutable_run_dir": str(run_dir.resolve())}
    )
    write_json(metadata_path(model_path), metadata)
    write_json(
        Path(report_dir) / "active_run.json", {**manifest, "run_dir": str(run_dir.resolve())}
    )
