"""Isolated fixture archives only; never publish the artifacts built by these tests."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from heart_disease.bundle import (
    build_bundle,
    checksum,
    extract_bundle,
    safe_member,
    sanitize,
    verify_directory,
)
from heart_disease.config import FEATURES, ROOT, UCI_URL
from heart_disease.demo import public_demo_record
from heart_disease.modeling import build_models, dependency_versions
from heart_disease.prediction import load_artifact, save_artifact


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "C:/drive",
        "models/../../outside",
        "models\\bad",
        "src//module.py",
        "src/module.py.",
        "src/module.py ",
        "src/NUL.py",
        "src/COM1",
        "src/invalid?.py",
        "src/\x00module.py",
    ],
)
def test_archive_paths_cannot_escape(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_archive_sha_and_zip_slip_are_rejected(tmp_path):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("../outside.txt", "test")
    payload = stream.getvalue()
    with pytest.raises(ValueError, match="SHA-256"):
        extract_bundle(payload, tmp_path / "bad_sha", "0" * 64)
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(payload, tmp_path / "bad_path", checksum(payload))
    assert not (tmp_path / "outside.txt").exists()


def test_bundle_deterministic_portable_and_tamper_checked(dataset, tmp_path):
    pipeline = build_models()[0]["Logistic Regression"].fit(
        dataset.frame[FEATURES], dataset.frame.target
    )
    source = {
        **dataset.metadata,
        "source": {"url": UCI_URL, "source": "Isolated official-source fixture mock"},
    }
    model = tmp_path / "test_only/model.joblib"
    save_artifact(
        pipeline,
        {
            "run_id": "unit-only",
            "selected_model": "Logistic Regression",
            "data": source,
            "dependency_versions": dependency_versions(),
        },
        model,
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "example.json").write_text(json.dumps({"path": "C:\\Users\\private\\source.csv"}))
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    a = build_bundle(ROOT, model, reports, first)
    b = build_bundle(ROOT, model, reports, second)
    assert a["sha256"] == b["sha256"]
    target = tmp_path / "moved/portable_app"
    manifest = extract_bundle(first.read_bytes(), target, a["sha256"])
    assert not any(name.startswith("data/raw/") for name in manifest["files"])
    restored, metadata = load_artifact(target / "models/heart_model.joblib")
    record, label = public_demo_record(metadata, 1)
    assert set(record) == set(FEATURES) and "Public UCI" in label
    assert restored.predict_proba(dataset.frame[FEATURES].iloc[:1]).shape == (1, 2)
    assert "C:\\Users\\private" not in (target / "reports/example.json").read_text()
    (target / "app.py").write_text("damaged")
    with pytest.raises(ValueError, match="damaged"):
        verify_directory(target)


def test_sanitizer_removes_runtime_paths():
    result = sanitize(
        {"_artifact_dir": "ignored", "path": "C:\\Users\\private\\source.csv"}, Path("/example")
    )
    assert "_artifact_dir" not in result
    assert result["path"] == "[training workspace]/source.csv"
