"""AppTest exercises navigation and recovery without a network or real data."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import heart_disease.config as config
from heart_disease.config import FEATURES, UCI_URL
from heart_disease.modeling import build_models, dependency_versions
from heart_disease.prediction import save_artifact


def test_all_pages_work_without_data_or_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(config, "DATA_PATH", tmp_path / "missing.data")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "no_reports")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    for name in [
        "Patient Input & Prediction",
        "Model Evaluation",
        "Explainability",
        "About & Limitations",
    ]:
        app.sidebar.radio[0].set_value(name).run(timeout=30)
        assert not app.exception, name
    assert any("Not clinically validated" in warning.value for warning in app.warning)


def test_demo_reset_and_export_controls_with_isolated_fixture(dataset, tmp_path, monkeypatch):
    pipeline = build_models()[0]["Logistic Regression"].fit(
        dataset.frame[FEATURES], dataset.frame.target
    )
    model = tmp_path / "unit_model.joblib"
    # Mock provenance only within this isolated UI test, never real evidence.
    data = {**dataset.metadata, "source": {"url": UCI_URL, "source": "Isolated test fixture"}}
    save_artifact(
        pipeline,
        {
            "run_id": "unit-only",
            "selected_model": "Logistic Regression",
            "probability_method": "raw",
            "data": data,
            "dependency_versions": dependency_versions(),
        },
        model,
    )
    monkeypatch.setattr(config, "MODEL_PATH", model)
    monkeypatch.setattr(config, "DATA_PATH", tmp_path / "unused.data")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "no_reports")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value("Patient Input & Prediction").run(timeout=30)
    assert not app.exception
    app.button(key="load_demo").click().run(timeout=30)
    assert not app.exception and not app.error
    assert all(widget.value is not None for widget in app.number_input)
    app.button(key="predict_submit").click().run(timeout=30)
    assert not app.exception and not app.error
    assert set(app.session_state["prediction_export"]).isdisjoint(FEATURES)
    app.button(key="reset_inputs").click().run(timeout=30)
    assert not app.exception
    assert all(widget.value is None for widget in app.number_input)
