import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from heart_disease.config import FEATURES
from heart_disease.experiments import (
    complete_run,
    create_run,
    preserve_existing,
    publish_run,
    verify_run_snapshot,
)
from heart_disease.io import read_json, sha256, write_json
from heart_disease.modeling import build_models
from heart_disease.prediction import metadata_path, save_artifact


def test_snapshot_and_publish_are_versioned_and_preserve_old_outputs(dataset, tmp_path):
    model = tmp_path / "active/model.joblib"
    reports = tmp_path / "active/reports"
    reports.mkdir(parents=True)
    (reports / "unit_report.txt").write_text("original isolated fixture report")
    pipeline = build_models()[0]["Logistic Regression"].fit(
        dataset.frame[FEATURES], dataset.frame.target
    )
    save_artifact(pipeline, {"run_id": "fixture01"}, model)
    runs = tmp_path / "runs"
    snapshot = preserve_existing(model, reports, runs)
    assert (snapshot / "reports/unit_report.txt").read_text() == "original isolated fixture report"
    assert preserve_existing(model, reports, runs) == snapshot
    run = create_run(runs)
    assert run != create_run(runs)
    shutil.copy2(model, run / "models/heart_model.joblib")
    shutil.copy2(metadata_path(model), run / "models/heart_model.metadata.json")
    (run / "reports/unit_report.txt").write_text("new isolated fixture report")
    with pytest.raises(FileNotFoundError):
        publish_run(run, model, reports)
    report = {
        "run_id": "fixture02",
        "created_utc": "unit test",
        "selected_model": "Logistic Regression",
        "data": {"sha256": "test-only"},
    }
    save_artifact(pipeline, report, run / "models/heart_model.joblib")
    write_json(run / "reports/evaluation_report.json", report)
    complete_run(run, report)
    publish_run(run, model, reports)
    assert (reports / "unit_report.txt").read_text() == "new isolated fixture report"
    assert (snapshot / "reports/unit_report.txt").read_text() == "original isolated fixture report"
    assert read_json(reports / "active_run.json")["run_id"] == "fixture02"


@pytest.fixture
def completed_run(dataset, tmp_path):
    run = create_run(tmp_path / "runs")
    report = {
        "run_id": "isolatedfixture",
        "created_utc": "unit test",
        "selected_model": "Logistic Regression",
        "data": {"sha256": dataset.metadata["sha256"]},
    }
    pipeline = build_models()[0]["Logistic Regression"].fit(
        dataset.frame[FEATURES], dataset.frame.target
    )
    save_artifact(pipeline, report, run / "models/heart_model.joblib")
    write_json(run / "reports/evaluation_report.json", report)
    (run / "reports/fixture.txt").write_text("isolated report")
    complete_run(run, report)
    return run


@pytest.mark.parametrize("damage", ["added", "removed", "report", "metadata", "model"])
def test_damaged_snapshot_cannot_change_active_outputs(completed_run, tmp_path, damage):
    active_model = tmp_path / "active/model.joblib"
    active_reports = tmp_path / "active/reports"
    active_reports.mkdir(parents=True)
    active_model.write_bytes(b"existing active model")
    (active_reports / "fixture.txt").write_text("existing active report")
    paths = {
        "added": completed_run / "reports/unlisted.txt",
        "removed": completed_run / "reports/fixture.txt",
        "report": completed_run / "reports/fixture.txt",
        "metadata": completed_run / "models/heart_model.metadata.json",
        "model": completed_run / "models/heart_model.joblib",
    }
    if damage == "removed":
        paths[damage].unlink()
    else:
        paths[damage].write_bytes(b"damaged fixture")
    with pytest.raises(ValueError, match="integrity|inventory"):
        publish_run(completed_run, active_model, active_reports)
    assert active_model.read_bytes() == b"existing active model"
    assert (active_reports / "fixture.txt").read_text() == "existing active report"
    assert not (active_reports / "active_run.json").exists()
    assert not (active_reports / "unlisted.txt").exists()


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "a\\b", "./report", "a//b"])
def test_unsafe_manifest_paths_are_rejected(completed_run, name):
    manifest = read_json(completed_run / "run.json")
    manifest["reports_sha256"][name] = "0" * 64
    write_json(completed_run / "run.json", manifest)
    with pytest.raises(ValueError, match="Unsafe report path"):
        verify_run_snapshot(completed_run)


def test_legacy_manifest_checks_identity_without_loading_model(completed_run, monkeypatch):
    manifest = read_json(completed_run / "run.json")
    manifest.pop("schema")
    manifest.pop("metadata_sha256")
    write_json(completed_run / "run.json", manifest)

    def forbidden_load(*args, **kwargs):
        pytest.fail("Snapshot verification must never deserialize a model")

    monkeypatch.setattr("joblib.load", forbidden_load)
    assert verify_run_snapshot(completed_run)["run_id"] == "isolatedfixture"
    metadata_file = completed_run / "models/heart_model.metadata.json"
    metadata = read_json(metadata_file)
    metadata["selected_model"] = "different model"
    write_json(metadata_file, metadata)
    with pytest.raises(ValueError, match="identity differs"):
        verify_run_snapshot(completed_run)


def test_report_identity_must_match_even_when_checksum_is_updated(completed_run):
    report_file = completed_run / "reports/evaluation_report.json"
    report = read_json(report_file)
    report["run_id"] = "differentfixture"
    write_json(report_file, report)
    manifest = read_json(completed_run / "run.json")
    manifest["reports_sha256"]["evaluation_report.json"] = sha256(report_file)
    write_json(completed_run / "run.json", manifest)
    with pytest.raises(ValueError, match="different runs"):
        verify_run_snapshot(completed_run)


def test_completed_manifest_cannot_be_rewritten(completed_run):
    original = (completed_run / "run.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        complete_run(completed_run, read_json(completed_run / "reports/evaluation_report.json"))
    assert (completed_run / "run.json").read_bytes() == original


def test_activation_cannot_overwrite_its_snapshot(completed_run, tmp_path):
    with pytest.raises(ValueError, match="immutable experiment snapshot"):
        publish_run(completed_run, completed_run / "models/heart_model.joblib", tmp_path / "active")
    assert verify_run_snapshot(completed_run)["schema"] == 2


def test_stale_reports_are_removed_only_after_all_paths_are_checked(completed_run, tmp_path):
    model, reports = tmp_path / "active/model.joblib", tmp_path / "active/reports"
    reports.mkdir(parents=True)
    (reports / "stale.txt").write_text("previous fixture report")
    write_json(reports / "active_run.json", {"reports_sha256": {"stale.txt": "", "../bad": ""}})
    with pytest.raises(ValueError, match="Unsafe report path"):
        publish_run(completed_run, model, reports)
    assert (reports / "stale.txt").is_file()
    assert not model.exists()
    write_json(reports / "active_run.json", {"reports_sha256": {"stale.txt": ""}})
    publish_run(completed_run, model, reports)
    assert not (reports / "stale.txt").exists()
    assert (reports / "fixture.txt").read_text() == "isolated report"


@pytest.mark.skipif(sys.platform == "win32", reason="Creating symlinks requires Windows privileges")
def test_snapshot_rejects_report_directory_symlink(completed_run, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "record.txt").write_text("outside fixture")
    (completed_run / "reports/linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic links"):
        verify_run_snapshot(completed_run)


def test_snapshot_verifier_command_reports_success_and_failure(completed_run):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/verify_experiment.py"),
        str(completed_run),
    ]
    passed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert '"status": "passed"' in passed.stdout
    assert '"metadata_checksum_recorded": true' in passed.stdout
    (completed_run / "reports/fixture.txt").write_text("damaged")
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 1
    assert "integrity check failed" in failed.stderr
    assert "Traceback" not in failed.stderr
