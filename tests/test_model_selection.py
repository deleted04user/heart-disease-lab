"""Selection uses training folds only, with stable ties and explicit failures."""

import numpy as np
import pytest

from heart_disease import model_selection


class PipelineStub:
    def __init__(self, failure=False):
        self.failure = failure
        self.fit_rows = None

    def fit(self, X, y):
        if self.failure:
            raise RuntimeError("fit failed")
        self.fit_rows = X.index.tolist()
        return self


def test_training_only_splits_and_stable_tie_breaking(dataset, monkeypatch):
    X = dataset.frame.drop(columns="target")
    y = dataset.frame.target
    observed = []

    def score(pipeline, features, labels, *, cv, scoring, n_jobs, error_score):
        assert features is X and labels is y
        assert n_jobs == 1 and error_score == "raise"
        assert set(scoring) == set(model_selection.METRICS)
        observed.append(cv)
        for train, validation in cv:
            assert set(train).isdisjoint(validation)
            assert len(train) + len(validation) == len(X)
        return {"test_" + metric: np.array([0.75] * len(cv)) for metric in scoring}

    monkeypatch.setattr(model_selection, "cross_validate", score)
    models = {"Z": PipelineStub(), "A": PipelineStub()}
    rows, fitted, scores, splits, winner = model_selection.select_models(
        models, {}, X, y, folds=2, cv_repeats=2
    )
    assert winner == "A" and [row["model"] for row in rows] == ["A", "Z"]
    assert len(splits) == 4 and all(cv is splits for cv in observed)
    assert all(p.fit_rows == X.index.tolist() for p in fitted.values())
    assert scores["A"]["roc_auc"] == [0.75] * 4
    assert rows[0]["cv_roc_auc_std"] == 0


def test_required_failure_stops_optional_failure_is_recorded(dataset, monkeypatch):
    def score(*args, **kwargs):
        return {"test_" + metric: np.array([0.5, 0.5]) for metric in model_selection.METRICS}

    monkeypatch.setattr(model_selection, "cross_validate", score)
    X, y = dataset.frame.drop(columns="target"), dataset.frame.target
    unavailable = {}
    result = model_selection.select_models(
        {"baseline": PipelineStub(), "XGBoost": PipelineStub(True)},
        unavailable,
        X,
        y,
        folds=2,
        cv_repeats=1,
    )
    assert result[-1] == "baseline" and "fit failed" in unavailable["XGBoost"]
    with pytest.raises(RuntimeError, match="fit failed"):
        model_selection.select_models(
            {"baseline": PipelineStub(True)}, {}, X, y, folds=2, cv_repeats=1
        )
    with pytest.raises(RuntimeError, match="No model"):
        model_selection.select_models({}, {}, X, y, folds=2, cv_repeats=1)
    with pytest.raises(ValueError, match="positive"):
        model_selection.select_models({}, {}, X, y, folds=2, cv_repeats=0)


@pytest.mark.parametrize("metric", ["roc_auc", "recall", "precision"])
def test_metric_ranking_precedes_alphabetical_tie(dataset, monkeypatch, metric):
    weaker, stronger = PipelineStub(), PipelineStub()

    def score(pipeline, *args, **kwargs):
        return {
            "test_" + name: np.array([0.9 if pipeline is stronger and name == metric else 0.5] * 2)
            for name in model_selection.METRICS
        }

    monkeypatch.setattr(model_selection, "cross_validate", score)
    result = model_selection.select_models(
        {"A": weaker, "Z": stronger}, {}, dataset.frame, dataset.frame.target, folds=2, cv_repeats=1
    )
    assert result[-1] == "Z"
