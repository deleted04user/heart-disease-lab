import sys

import numpy as np
import pytest

from heart_disease.config import FEATURES
from heart_disease.data import Dataset, DataValidationError
from heart_disease.evaluation import evaluate, specificity
from heart_disease.modeling import build_models, split_dataset


def test_stratified_disjoint_split_reproducible(dataset):
    a = split_dataset(dataset)
    b = split_dataset(dataset)
    X_train, X_test, y_train, y_test, train_ids, test_ids, folds = a
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(len(dataset.frame)))
    np.testing.assert_array_equal(train_ids, b[4])
    assert y_train.nunique() == y_test.nunique() == 2
    assert list(X_train.columns) == list(X_test.columns) == FEATURES
    assert folds == 5


def test_folds_adapt_and_too_small_stops(dataset):
    small = Dataset(dataset.frame.iloc[:8], {})
    assert split_dataset(small)[-1] == 3
    with pytest.raises(DataValidationError):
        split_dataset(Dataset(dataset.frame.iloc[:4], {}), test_size=0.5)


def test_available_models_fit_unit_fixture_only(dataset):
    models, unavailable = build_models()
    assert {"Logistic Regression", "Random Forest"}.issubset(models)
    assert "XGBoost" in models or "XGBoost" in unavailable
    for pipeline in models.values():
        pipeline.fit(dataset.frame[FEATURES], dataset.frame.target)
        probability = pipeline.predict_proba(dataset.frame[FEATURES])
        assert probability.shape == (len(dataset.frame), 2)
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
        assert set(pipeline.classes_) == {0, 1}


def test_metrics_have_correct_confusion_semantics():
    result = evaluate([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    for metric in ["accuracy", "precision", "recall", "f1", "specificity"]:
        assert result[metric] == 0.5
    assert result["roc_auc"] == 0.75
    assert evaluate([0, 0], [0, 0], [0.1, 0.2])["roc_auc"] is None
    assert specificity([0, 0], [0, 0]) == 1.0


def test_xgboost_dependency_failure_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    models, unavailable = build_models()
    assert set(models) == {"Logistic Regression", "Random Forest"}
    assert "import failure" in unavailable["XGBoost"]
