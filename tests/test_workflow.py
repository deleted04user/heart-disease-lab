"""End-to-end checks on artificial isolated fixtures, never experiment evidence."""

import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

import heart_disease.config as config
import heart_disease.modeling as modeling
from heart_disease.config import FEATURES, UCI_URL
from heart_disease.eda import run_eda
from heart_disease.evaluation import evaluate
from heart_disease.prediction import load_artifact
from heart_disease.transfer import investigate


def test_training_transfer_reports_and_populated_dashboard(dataset, frame, tmp_path, monkeypatch):
    # Mock public provenance applies only inside this temporary test directory.
    dataset.metadata["source"] = {"url": UCI_URL, "source": "Artificial isolated test fixture"}
    models, _ = modeling.build_models()
    models.pop("XGBoost", None)
    models["Random Forest"].set_params(classifier__n_estimators=12)
    monkeypatch.setattr(
        modeling,
        "build_models",
        lambda: (models, {"XGBoost": "Disabled only in this isolated integration fixture"}),
    )
    external = frame.copy()
    external["age"] += 1
    external_path = tmp_path / "external_fixture.csv"
    external.to_csv(external_path, index=False)
    reports, model = tmp_path / "reports", tmp_path / "models/heart_model.joblib"
    run_eda(dataset, reports)
    report = modeling.train(
        dataset,
        model_path=model,
        report_dir=reports,
        cv_repeats=1,
        bootstrap_resamples=100,
        importance_repeats=2,
        external_path=external_path,
        historical_holdout=False,
    )
    train_ids, test_ids = (report["partition_indices"][key] for key in ("train", "test"))
    assert set(train_ids).isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == len(frame)
    assert report["external"]["status"] == "evaluated"
    pipeline, metadata = load_artifact(model)
    X_test, y_test = dataset.frame.iloc[test_ids][FEATURES], dataset.frame.iloc[test_ids].target
    reproduced = evaluate(y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1])
    assert reproduced == report["test_results"][report["selected_model"]]
    assert metadata["run_id"] == report["run_id"]
    cv_rows = json.loads((reports / "evaluation_report.json").read_text())["cv_results"]
    unavailable = next(row for row in cv_rows if row["model"] == "XGBoost")
    assert unavailable["cv_roc_auc_mean"] is None
    assert all(not any(key.startswith("test_") for key in row) for row in cv_rows)

    original_predictions = pipeline.predict_proba(X_test)
    diagnostics = investigate(model, reports)
    assert diagnostics["deployment_model_changed"] is False
    assert diagnostics["external_rows_used_for_fitting_or_selection"] == 0
    assert len(diagnostics["training_only_stress"]) == 3
    reloaded, _ = load_artifact(model)
    np.testing.assert_array_equal(original_predictions, reloaded.predict_proba(X_test))

    monkeypatch.setattr(config, "MODEL_PATH", model)
    monkeypatch.setattr(config, "DATA_PATH", Path(dataset.metadata["path"]))
    monkeypatch.setattr(config, "REPORT_DIR", reports)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    for page in ["Project Overview", "Model Evaluation", "Explainability", "About & Limitations"]:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception and not app.error, page
    app.sidebar.radio[0].set_value("Patient Input & Prediction").run(timeout=30)
    app.button(key="load_demo").click().run(timeout=30)
    app.button(key="predict_submit").click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["prediction_export"]["public_example_loaded"] is True
