"""Training-only cross-validation and deterministic base-model ranking."""

import logging

from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from .config import SEED
from .evaluation import specificity

LOG = logging.getLogger(__name__)
METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]


def select_models(models, unavailable, X_train, y_train, *, folds, cv_repeats):
    """Fit/rank supplied pipelines using training data only; return reusable CV splits."""
    if cv_repeats < 1:
        raise ValueError("cv_repeats must be positive.")
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=cv_repeats, random_state=SEED)
    splits = list(cv.split(X_train, y_train))
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "specificity": make_scorer(specificity),
    }
    rows, fitted = [], {}
    fold_scores: dict[str, dict[str, list[float]]] = {}
    for name, pipeline in models.items():
        LOG.info("Training-only %d × %d-fold CV: %s", cv_repeats, folds, name)
        try:
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=splits,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
            row = {
                "model": name,
                "status": "trained",
                "cv_folds": folds,
                "cv_repeats": cv_repeats,
                "reason": "",
                "probability_method": "raw",
            }
            fold_scores[name] = {}
            for metric in METRICS:
                values = scores[f"test_{metric}"]
                row[f"cv_{metric}_mean"] = float(values.mean())
                row[f"cv_{metric}_std"] = float(values.std(ddof=0))
                fold_scores[name][metric] = values.tolist()
            pipeline.fit(X_train, y_train)
            fitted[name] = pipeline
            rows.append(row)
        except Exception as exc:
            # Platform/import/training failure in optional XGBoost is explicit;
            # failures in required baselines are code/data defects and stop the run.
            if name != "XGBoost":
                raise
            unavailable[name] = f"CV or fitting failed: {type(exc).__name__}: {exc}"
    if not rows:
        raise RuntimeError("No model could be trained.")
    # Winner is frozen here, before ANY final test evaluation.
    rows.sort(
        key=lambda row: (
            -row["cv_roc_auc_mean"],
            -row["cv_recall_mean"],
            -row["cv_precision_mean"],
            row["model"],
        )
    )
    selected = rows[0]["model"]
    return rows, fitted, fold_scores, splits, selected
