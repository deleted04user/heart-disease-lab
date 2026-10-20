"""Fixed, modest models compared on training folds; test labels never rank models."""

import importlib.metadata
import logging
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    BOUNDS,
    CATEGORY_LABELS,
    FEATURES,
    MODEL_PATH,
    REPORT_DIR,
    SEED,
    TARGET,
    TEST_SIZE,
)
from .data import Dataset, DataValidationError
from .evaluation import evaluate, save_cv_plot, save_diagnostic_plots
from .explainability import explain_model
from .io import write_json
from .model_reports import _write_model_card, _write_project_report
from .model_selection import METRICS, select_models
from .prediction import save_artifact
from .preprocessing import build_preprocessor
from .research import (
    compare_calibration,
    probability_scores,
    reliability_bins,
    save_reliability_plot,
    save_uncertainty_plot,
    subgroup_results,
    uncertainty,
)

LOG = logging.getLogger(__name__)


def build_models() -> tuple[dict[str, Pipeline], dict[str, str]]:
    classifiers = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=1,
        ),
    }
    unavailable = {}
    try:
        from xgboost import XGBClassifier

        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.04,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
        )
    except (ImportError, OSError) as exc:
        unavailable["XGBoost"] = f"Dependency/platform import failure: {type(exc).__name__}: {exc}"
    return {
        name: Pipeline(
            [
                ("preprocess", build_preprocessor(scale=name == "Logistic Regression")),
                ("classifier", model),
            ]
        )
        for name, model in classifiers.items()
    }, unavailable


def split_dataset(dataset: Dataset, test_size: float = TEST_SIZE):
    X, y = dataset.frame[FEATURES], dataset.frame[TARGET]
    if not 0 < test_size < 1:
        raise DataValidationError("test_size must be strictly between 0 and 1.")
    try:
        train_ids, test_ids = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=SEED, stratify=y
        )
    except ValueError as exc:
        raise DataValidationError(f"Stratified holdout is not possible: {exc}") from exc
    y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
    if y_test.nunique() != 2 or y_train.nunique() != 2 or y_train.value_counts().min() < 2:
        raise DataValidationError(
            "Insufficient class counts: test must contain both classes and training needs >=2 per class for CV. Supply more data or an explicitly chosen --test-size."
        )
    folds = min(5, int(y_train.value_counts().min()))
    return X.iloc[train_ids], X.iloc[test_ids], y_train, y_test, train_ids, test_ids, folds


def dependency_versions() -> dict:
    versions = {}
    for name in [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "streamlit",
        "joblib",
        "xgboost",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def train(
    dataset: Dataset,
    *,
    model_path: Path = MODEL_PATH,
    report_dir: Path = REPORT_DIR,
    test_size: float = TEST_SIZE,
    importance_repeats: int = 20,
    cv_repeats: int = 3,
    bootstrap_resamples: int = 1000,
    external_path: Path | None = None,
    historical_holdout: bool = True,
) -> dict:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, train_ids, test_ids, folds = split_dataset(dataset, test_size)
    models, unavailable = build_models()
    rows, fitted, fold_scores, splits, selected = select_models(
        models, unavailable, X_train, y_train, folds=folds, cv_repeats=cv_repeats
    )
    parameters = {
        name: {
            key: (
                "NaN (default missing marker)"
                if isinstance(value, float) and not np.isfinite(value)
                else value
            )
            for key, value in fitted[name].named_steps["classifier"].get_params().items()
        }
        for name in fitted
    }
    LOG.info(
        "Base classifier selected from repeated training CV: %s. Comparing probability methods inside training only.",
        selected,
    )
    calibration, final_pipeline = compare_calibration(
        models[selected], X_train, y_train, splits, folds=folds, repeats=cv_repeats
    )
    fitted[selected] = final_pipeline
    LOG.info(
        "Probability method frozen: %s. Evaluating historical audit holdout.",
        calibration["selected_method"],
    )
    test_metrics, probabilities = {}, {}
    for row in rows:
        name = row["model"]
        pipeline = fitted[name]
        probability = pipeline.predict_proba(X_test)[:, 1]
        test_metrics[name] = evaluate(y_test, pipeline.predict(X_test), probability)
        probabilities[name] = probability
        row.update({f"test_{metric}": test_metrics[name][metric] for metric in METRICS})
        row["selected"] = name == selected
        row["probability_method"] = calibration["selected_method"] if name == selected else "raw"
        row["test_role"] = (
            "historical audit; previously inspected"
            if historical_holdout
            else "held-out descriptive evaluation"
        )
    for name, reason in unavailable.items():
        LOG.warning("%s unavailable: %s", name, reason)
        rows.append({"model": name, "status": "unavailable", "reason": reason, "selected": False})
    comparison = pd.DataFrame(rows)
    run_id = uuid.uuid4().hex
    configuration = {
        "seed": SEED,
        "test_fraction": test_size,
        "cv_folds": folds,
        "cv_repeats": cv_repeats,
        "cv_total_splits": len(splits),
        "cv_shuffle": True,
        "cv_std_ddof": 0,
        "threshold": 0.5,
        "threshold_tuned": False,
        "hyperparameter_search": False,
        "synthetic_oversampling": False,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "historical_holdout": historical_holdout,
        "bootstrap_resamples": bootstrap_resamples,
        "selection": "Highest training repeated-CV mean ROC-AUC for the base classifier; ties: recall, precision, then name. Probability method selected separately with nested training-only CV. Neither holdout nor external results select models or calibrators.",
    }
    report = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected,
        "data": dataset.metadata,
        "configuration": configuration,
        "cv_results": [],
        "cv_fold_scores": fold_scores,
        "test_results": test_metrics,
        "unavailable": unavailable,
        "model_parameters": parameters,
        "partition_indices": {"train": train_ids.tolist(), "test": test_ids.tolist()},
        "class_counts": {
            "train": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().items()},
        },
        "limitations": [
            "Small, historical observational dataset; no external or prospective validation.",
            "Test performance is uncertain with so few observations; fold SD is not a confidence interval.",
            "Probabilities are uncalibrated model outputs, not clinical risk scores.",
            "Secondary test comparisons and held-out importance are descriptive; do not tune using them.",
            "Source selection, missingness, limited demographic encoding and hospital sampling can bias results.",
        ],
    }
    # Convert unavailable NaN cells to real JSON nulls, never non-standard NaN.
    report["cv_results"] = [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in comparison[[c for c in comparison if not str(c).startswith("test_")]].to_dict(
            orient="records"
        )
    ]
    report["calibration"] = calibration
    selected_probability = probabilities[selected]
    selected_predictions = fitted[selected].predict(X_test)
    report["uncertainty"] = uncertainty(
        y_test, selected_predictions, selected_probability, resamples=bootstrap_resamples
    )
    report["subgroups"] = subgroup_results(
        X_test, y_test, selected_predictions, selected_probability
    )
    report["probability_scores"] = probability_scores(y_test, selected_probability)
    report["holdout_reliability"] = reliability_bins(y_test, selected_probability)
    report["test_interpretation"] = (
        "Previously inspected Cleveland holdout retained as an audit. Updated results are not a fresh unbiased final benchmark."
        if historical_holdout
        else "Descriptive holdout evaluation; no clinical claims."
    )
    report["external"] = {
        "status": "unavailable",
        "reason": "No external file supplied. Run scripts/download_external.py and train with --external data/raw/processed.hungarian.data.",
    }
    if external_path is not None:
        from .external import evaluate_external

        LOG.info("Evaluating frozen pipeline on explicit external cohort: %s", external_path)
        try:
            report["external"] = evaluate_external(
                fitted[selected], dataset, external_path, resamples=bootstrap_resamples
            )
        except (ValueError, OSError) as exc:
            report["external"] = {
                "status": "unavailable",
                "reason": f"External validation/audit failed: {type(exc).__name__}: {exc}",
            }
            LOG.warning(report["external"]["reason"])
    report["limitations"] = [
        "Small historical observational cohorts; no prospective or clinical validation.",
        "Cleveland holdout has been inspected previously; new results are an audit, not a new final benchmark.",
        "Repeated folds are correlated; fold SD is not a confidence interval.",
        "Intervals condition on a fixed model and do not include training/selection or site uncertainty.",
        "Calibration is selected on training folds only; probability outputs are not clinically validated risk scores.",
        "Demographic slices are small and descriptive; no fairness or equivalence claim.",
        "External feature missingness and hospital selection can materially bias transfer results.",
        "Do not tune using viewed holdout, subgroup, importance or external diagnostics.",
    ]
    comparison.insert(0, "run_id", run_id)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    write_json(report_dir / "evaluation_report.json", report)
    write_json(report_dir / "data_provenance.json", dataset.metadata)
    write_json(report_dir / "calibration_report.json", {"run_id": run_id, **calibration})
    write_json(report_dir / "external_evaluation.json", {"run_id": run_id, **report["external"]})
    pd.DataFrame(report["subgroups"]).to_csv(report_dir / "subgroup_evaluation.csv", index=False)
    pd.DataFrame(calibration["fold_scores"]).to_csv(
        report_dir / "calibration_comparison.csv", index=False
    )
    # Each repeat average is shown separately; repeats are not treated as
    # independent observations for a confidence interval or significance test.
    repeat_rows = []
    for name, metrics in fold_scores.items():
        for repeat in range(cv_repeats):
            repeat_rows.append(
                {
                    "model": name,
                    "repeat": repeat + 1,
                    **{
                        metric: float(np.mean(values[repeat * folds : (repeat + 1) * folds]))
                        for metric, values in metrics.items()
                    },
                }
            )
    pd.DataFrame(repeat_rows).to_csv(report_dir / "cv_repeat_summary.csv", index=False)
    save_uncertainty_plot(report["uncertainty"], report_dir / "figures/metric_intervals.png")
    save_reliability_plot(
        calibration["reliability"],
        report_dir / "figures/training_reliability.png",
        title="Training out-of-fold probability reliability",
    )
    save_reliability_plot(
        {"Cleveland audit": report["holdout_reliability"]},
        report_dir / "figures/holdout_reliability.png",
        title="Previously inspected Cleveland audit holdout",
    )
    if report["external"]["status"] == "evaluated":
        pd.DataFrame(report["external"]["subgroups"]).to_csv(
            report_dir / "external_subgroups.csv", index=False
        )
        save_reliability_plot(
            {"Hungarian external cohort": report["external"]["reliability"]},
            report_dir / "figures/external_reliability.png",
            title="Frozen pipeline on the separate Hungarian site",
        )
    save_diagnostic_plots(test_metrics, probabilities, y_test, selected, report_dir / "figures")
    save_cv_plot(comparison, report_dir / "figures/cv_comparison.png")
    importance = explain_model(
        fitted[selected], X_test, y_test, report_dir, repeats=importance_repeats
    )
    write_json(
        report_dir / "explainability.json",
        {
            "run_id": run_id,
            "method": "Original-feature permutation importance on final test set; ROC-AUC decrease",
            "repeats": importance_repeats,
            "seed": SEED,
            "test_rows": len(X_test),
            "features": importance.to_dict(orient="records"),
            "caution": "Descriptive only; correlated features share importance. Negative importance can occur. Repeat SD is not sampling uncertainty. No causal or clinical interpretation.",
        },
    )
    metadata = {
        "run_id": run_id,
        "created_utc": report["created_utc"],
        "selected_model": selected,
        "probability_method": calibration["selected_method"],
        "data": dataset.metadata,
        "training_configuration": configuration,
        "model_parameters": parameters[selected],
        "dependency_versions": dependency_versions(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "evaluation": test_metrics[selected],
        "input_schema": {
            "numeric_bounds": BOUNDS,
            "categorical_codes": {col: list(codes) for col, codes in CATEGORY_LABELS.items()},
            "missing_prediction_inputs": "rejected",
        },
        "report_dir": str(report_dir),
    }
    save_artifact(fitted[selected], metadata, model_path)
    _write_project_report(report, comparison, importance, report_dir)
    _write_model_card(report, report_dir)
    return report
