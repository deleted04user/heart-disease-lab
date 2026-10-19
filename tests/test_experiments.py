import shutil

import pytest

from heart_disease.config import FEATURES
from heart_disease.experiments import complete_run, create_run, preserve_existing, publish_run
from heart_disease.io import read_json
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
    save_artifact(pipeline, {"run_id": "fixture02"}, run / "models/heart_model.joblib")
    complete_run(
        run,
        {
            "run_id": "fixture02",
            "created_utc": "unit test",
            "selected_model": "Logistic Regression",
            "data": {"sha256": "test-only"},
        },
    )
    publish_run(run, model, reports)
    assert (reports / "unit_report.txt").read_text() == "new isolated fixture report"
    assert (snapshot / "reports/unit_report.txt").read_text() == "original isolated fixture report"
    assert read_json(reports / "active_run.json")["run_id"] == "fixture02"
