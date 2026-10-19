"""Artificial records below are isolated unit fixtures, never experimental evidence."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold

from heart_disease.config import FEATURES
from heart_disease.modeling import build_models
from heart_disease.prediction import load_artifact, save_artifact
from heart_disease.research import (
    compare_calibration,
    make_probability_pipeline,
    probability_scores,
    reliability_bins,
    subgroup_results,
    uncertainty,
    wilson_interval,
)


def test_wilson_known_counts_and_undefined():
    result = wilson_interval(25, 100)
    assert result["lower"] == pytest.approx(0.175452, abs=1e-5)
    assert result["upper"] == pytest.approx(0.343045, abs=1e-5)
    assert wilson_interval(0, 0)["estimate"] is None
    assert wilson_interval(0, 10)["lower"] == 0
    assert wilson_interval(10, 10)["upper"] == 1
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_bootstrap_reproducible_perfect_auc_and_single_class():
    y, predicted, probability = [0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.2, 0.8, 0.9]
    a = uncertainty(y, predicted, probability, resamples=100)
    assert a == uncertainty(y, predicted, probability, resamples=100)
    assert a["metrics"]["roc_auc"]["lower"] == a["metrics"]["roc_auc"]["upper"] == 1
    assert a["metrics"]["accuracy"]["estimate"] == 0.75
    single = uncertainty([0, 0], [0, 0], [0.1, 0.2], resamples=100)
    assert single["metrics"]["roc_auc"]["estimate"] is None
    assert single["metrics"]["recall"]["estimate"] is None
    with pytest.raises(ValueError):
        uncertainty(y, predicted, probability, resamples=1)


def test_reliability_edges_empty_bins_and_counts():
    bins = reliability_bins([0, 1, 1], [0, 0.5, 1], bins=5)
    assert sum(row["n"] for row in bins) == 3
    assert bins[0]["mean_probability"] == 0
    assert bins[-1]["mean_probability"] == 1
    assert bins[1]["positive_fraction"] is None
    assert probability_scores([0, 1], [0.1, 0.9])["brier"] == pytest.approx(0.01)


def test_calibration_is_nested_and_rule_controls_retention(dataset):
    X, y = dataset.frame[FEATURES], dataset.frame.target
    model = build_models()[0]["Logistic Regression"]
    candidate = make_probability_pipeline(model, "sigmoid")
    calibrated = candidate.named_steps["classifier"]
    assert set(calibrated.estimator.named_steps) == {"preprocess", "classifier"}
    with pytest.raises(ValueError):
        make_probability_pipeline(model, "isotonic")
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
    report, fitted = compare_calibration(model, X, y, list(cv.split(X, y)), folds=2, repeats=2)
    summary = {row["method"]: row for row in report["summary"]}
    retain = (
        summary["raw"]["brier_mean"] - summary["sigmoid"]["brier_mean"] >= 0.002
        and summary["sigmoid"]["log_loss_mean"] < summary["raw"]["log_loss_mean"]
    )
    assert report["selected_method"] == ("sigmoid" if retain else "raw")
    assert len(report["fold_scores"]) == 8
    assert all(sum(row["n"] for row in curve) == len(X) for curve in report["reliability"].values())
    assert fitted.predict_proba(X).shape == (len(X), 2)
    assert not hasattr(model.named_steps["classifier"], "classes_")


def test_subgroup_undefined_denominators_and_empty_groups():
    X = pd.DataFrame({"age": [35, 40], "sex": [0, 0]})
    rows = subgroup_results(X, [0, 0], [0, 0], [0.1, 0.2])
    female = next(row for row in rows if row["group"] == "Female (0)")
    assert female["n"] == 2 and female["recall"] is None and female["precision"] is None
    assert female["roc_auc"] is None
    male = next(row for row in rows if row["group"] == "Male (1)")
    assert male["n"] == 0 and male["accuracy"] is None


def test_calibrated_complete_pipeline_survives_persistence(dataset, tmp_path):
    X, y = dataset.frame[FEATURES], dataset.frame.target
    pipeline = make_probability_pipeline(build_models()[0]["Logistic Regression"], "sigmoid").fit(
        X, y
    )
    path = tmp_path / "unit_calibrated.joblib"
    save_artifact(
        pipeline,
        {
            "run_id": "test-only",
            "selected_model": "Logistic Regression",
            "probability_method": "sigmoid",
        },
        path,
    )
    restored, _ = load_artifact(path)
    np.testing.assert_allclose(pipeline.predict_proba(X), restored.predict_proba(X))
