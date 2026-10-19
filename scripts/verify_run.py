"""Verify actual saved experiment, source hash, UI pages, and real-record form inference."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from _common import ROOT
from heart_disease.config import FEATURES, MODEL_PATH, REPORT_DIR
from heart_disease.data import load_dataset
from heart_disease.evaluation import evaluate
from heart_disease.io import read_json
from heart_disease.prediction import load_artifact, predict_records


def verify(server_url: str | None = None, *, require_external: bool = False) -> dict:
    pipeline, metadata = load_artifact(MODEL_PATH)
    report = read_json(REPORT_DIR / "evaluation_report.json")
    if require_external:
        assert report.get("external", {}).get("status") == "evaluated", (
            "A successful external evaluation is required"
        )
    assert metadata["run_id"] == report["run_id"], "Model/report mismatch"
    data = report["data"]
    source_path = Path(data["path"])
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    dataset = load_dataset(
        source_path,
        schema=data["schema"],
        target_semantics=data["target_semantics"],
        site=data.get("site") or "Cleveland",
    )
    assert dataset.metadata["sha256"] == data["sha256"], "Source changed after training"
    train_ids = report["partition_indices"]["train"]
    test_ids = report["partition_indices"]["test"]
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(len(dataset.frame)))
    holdout = dataset.frame.iloc[test_ids]
    X, y = holdout[FEATURES], holdout.target
    recomputed = evaluate(y, pipeline.predict(X), pipeline.predict_proba(X)[:, 1])
    stored = report["test_results"][report["selected_model"]]
    for metric in ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]:
        assert np.isclose(recomputed[metric], stored[metric]), metric
    assert recomputed["confusion_matrix"] == stored["confusion_matrix"]
    assert metadata["evaluation"] == stored
    comparison = pd.read_csv(REPORT_DIR / "model_comparison.csv")
    assert set(comparison.run_id) == {metadata["run_id"]}
    trained = comparison[comparison.status == "trained"]
    expected_winner = (
        trained.sort_values(
            ["cv_roc_auc_mean", "cv_recall_mean", "cv_precision_mean", "model"],
            ascending=[False, False, False, True],
        )
        .iloc[0]
        .model
    )
    assert expected_winner == report["selected_model"]
    for path in REPORT_DIR.glob("*.json"):
        with path.open(encoding="utf-8") as handle:
            json.load(
                handle,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"Invalid JSON constant: {token}")
                ),
            )
    for path in (REPORT_DIR / "figures").glob("*.png"):
        image = mpimg.imread(path)
        assert image.ndim == 3 and min(image.shape[:2]) > 100, path.name
    explanation = read_json(REPORT_DIR / "explainability.json")
    assert explanation["run_id"] == metadata["run_id"]
    assert set(pd.DataFrame(explanation["features"]).feature) == set(FEATURES)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception, str(app.exception)
    assert any(metric.value == str(data["rows"]) for metric in app.metric)
    for page in [
        "Model Evaluation",
        "Explainability",
        "About & Limitations",
        "Patient Input & Prediction",
    ]:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, f"{page}: {app.exception}"
    app.button(key="predict_submit").click().run(timeout=30)
    assert any("Complete all fields" in error.value for error in app.error)
    # A public source record checks wiring only; it never changes selection/metrics.
    record = dataset.frame[FEATURES].dropna().iloc[0]
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        app.number_input(key=col).set_value(
            float(record[col]) if col == "oldpeak" else int(record[col])
        )
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        app.selectbox(key=col).set_value(int(record[col]))
    app.button(key="predict_submit").click().run(timeout=30)
    assert not app.exception and not app.error, str(app.exception)
    expected = predict_records(pipeline, record.to_dict()).iloc[0]
    assert any(
        metric.value == f"{expected.positive_class_probability:.1%}" for metric in app.metric
    ), "UI probability differs from inference API"
    expected_class_text = "positive class (1)" if expected.prediction == 1 else "negative class (0)"
    assert any(expected_class_text in title.value for title in app.subheader)
    if "calibration" in report:
        assert (
            report["configuration"]["cv_total_splits"]
            == report["configuration"]["cv_folds"] * report["configuration"]["cv_repeats"]
        )
        assert report["calibration"]["selected_method"] == metadata["probability_method"]
        for entry in report["calibration"]["reliability"].values():
            assert sum(bin["n"] for bin in entry) == len(train_ids)
        assert "age" not in app.session_state["prediction_export"]
        app.button(key="reset_inputs").click().run(timeout=30)
        assert not app.exception
        assert all(widget.value is None for widget in app.number_input)
        app.button(key="load_demo").click().run(timeout=30)
        assert not app.exception and not app.error
        assert all(widget.value is not None for widget in app.number_input)
        app.button(key="predict_submit").click().run(timeout=30)
        assert not app.exception and not app.error
        assert set(app.session_state["prediction_export"]).isdisjoint(FEATURES)
        external = report["external"]
        if external["status"] == "evaluated":
            from heart_disease.external import evaluate_external

            external_path = Path(external["data"]["path"])
            if not external_path.is_absolute():
                external_path = ROOT / external_path
            reproduced = evaluate_external(pipeline, dataset, external_path, resamples=100)
            assert reproduced["metrics"] == external["metrics"]
            assert external["fitting_on_external_data"] is False
    if server_url:
        with urlopen(server_url.rstrip("/") + "/_stcore/health", timeout=10) as response:
            assert response.read().strip() == b"ok", "Streamlit unhealthy"
    return {
        "run_id": report["run_id"],
        "real_records": len(dataset.frame),
        "models_trained": trained.model.tolist(),
        "verified": [
            "source checksum",
            "disjoint complete partitions",
            "reloaded holdout metrics",
            "CV winner",
            "JSON and image artifacts",
            "all populated UI pages",
            "blank form validation",
            "real-record form/API prediction consistency",
        ]
        + (
            [
                "repeated CV and calibration coherence",
                "public example and reset controls",
                "export excludes input measurements",
                "external metrics reproduced when available",
            ]
            if "calibration" in report
            else []
        )
        + (["server health"] if server_url else []),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default=None,
        help="Optionally verify an already-running local Streamlit server",
    )
    parser.add_argument(
        "--require-external",
        action="store_true",
        help="Fail unless the external cohort was actually evaluated",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.server_url, require_external=args.require_external), indent=2))
