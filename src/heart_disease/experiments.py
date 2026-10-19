"""Preserve immutable run folders and publish only a fully completed experiment."""

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT
from .io import read_json, sha256, write_json
from .prediction import metadata_path

RUNS_DIR = ROOT / "experiments/runs"


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
    model = run_dir / "models/heart_model.joblib"
    metadata = read_json(metadata_path(model))
    if metadata["run_id"] != report["run_id"] or metadata["model_sha256"] != sha256(model):
        raise ValueError("Model, metadata and report must describe the same completed run.")
    hashes = {
        str(path.relative_to(run_dir / "reports").as_posix()): sha256(path)
        for path in (run_dir / "reports").rglob("*")
        if path.is_file()
    }
    write_json(
        run_dir / "run.json",
        {
            "run_id": report["run_id"],
            "status": "complete",
            "created_utc": report["created_utc"],
            "selected_model": report["selected_model"],
            "probability_method": report.get("calibration", {}).get("selected_method", "raw"),
            "data_sha256": report["data"]["sha256"],
            "model_sha256": sha256(model),
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
    manifest = read_json(run_dir / "run.json")
    if manifest["status"] != "complete":
        raise ValueError("Only a completed run can be published.")
    source_model = run_dir / "models/heart_model.joblib"
    if sha256(source_model) != manifest["model_sha256"]:
        raise ValueError("Completed model checksum differs; publishing refused.")
    for name, checksum in manifest.get("reports_sha256", {}).items():
        source = (run_dir / "reports" / name).resolve()
        if not source.is_relative_to((run_dir / "reports").resolve()) or sha256(source) != checksum:
            raise ValueError("Completed report integrity check failed; publishing refused.")
    previous_manifest = Path(report_dir) / "active_run.json"
    if previous_manifest.exists():
        previous_files = set(read_json(previous_manifest).get("reports_sha256", {}))
        current_files = set(manifest.get("reports_sha256", {}))
        for name in previous_files - current_files:
            stale = (Path(report_dir) / name).resolve()
            if not stale.is_relative_to(Path(report_dir).resolve()):
                raise ValueError("Unsafe previous report path.")
            if stale.is_file():
                stale.unlink()
    for source in (run_dir / "reports").rglob("*"):
        if source.is_file():
            _copy_atomic(source, Path(report_dir) / source.relative_to(run_dir / "reports"))
    _copy_atomic(source_model, Path(model_path))
    metadata = read_json(run_dir / "models/heart_model.metadata.json")
    metadata.update(
        {"report_dir": str(Path(report_dir).resolve()), "immutable_run_dir": str(run_dir.resolve())}
    )
    write_json(metadata_path(model_path), metadata)
    write_json(
        Path(report_dir) / "active_run.json", {**manifest, "run_dir": str(run_dir.resolve())}
    )
