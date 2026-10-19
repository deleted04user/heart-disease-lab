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
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .config import FEATURES, MODEL_PATH, REPORT_DIR, SEED, TARGET, TEST_SIZE, BOUNDS, CATEGORY_LABELS
from .data import Dataset, DataValidationError
from .evaluation import evaluate, save_cv_plot, save_diagnostic_plots, specificity
from .explainability import explain_model
from .io import write_json
from .prediction import save_artifact
from .preprocessing import build_preprocessor
from .research import compare_calibration, probability_scores, reliability_bins, save_reliability_plot, save_uncertainty_plot, subgroup_results, uncertainty

LOG = logging.getLogger(__name__)
METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]


def build_models() -> tuple[dict[str, Pipeline], dict[str, str]]:
    classifiers = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=1),
    }
    unavailable = {}
    try:
        from xgboost import XGBClassifier
        classifiers["XGBoost"] = XGBClassifier(n_estimators=180, max_depth=2, learning_rate=0.04, min_child_weight=3, subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0, reg_alpha=0.1, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=1)
    except (ImportError, OSError) as exc:
        unavailable["XGBoost"] = f"Dependency/platform import failure: {type(exc).__name__}: {exc}"
    return {name: Pipeline([("preprocess", build_preprocessor(scale=name == "Logistic Regression")), ("classifier", model)]) for name, model in classifiers.items()}, unavailable


def split_dataset(dataset: Dataset, test_size: float = TEST_SIZE):
    X, y = dataset.frame[FEATURES], dataset.frame[TARGET]
    if not 0 < test_size < 1:
        raise DataValidationError("test_size must be strictly between 0 and 1.")
    try:
        train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=test_size, random_state=SEED, stratify=y)
    except ValueError as exc:
        raise DataValidationError(f"Stratified holdout is not possible: {exc}") from exc
    y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
    if y_test.nunique() != 2 or y_train.nunique() != 2 or y_train.value_counts().min() < 2:
        raise DataValidationError("Insufficient class counts: test must contain both classes and training needs >=2 per class for CV. Supply more data or an explicitly chosen --test-size.")
    folds = min(5, int(y_train.value_counts().min()))
    return X.iloc[train_ids], X.iloc[test_ids], y_train, y_test, train_ids, test_ids, folds


def dependency_versions() -> dict:
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "streamlit", "joblib", "xgboost"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def train(dataset: Dataset, *, model_path: Path = MODEL_PATH, report_dir: Path = REPORT_DIR, test_size: float = TEST_SIZE, importance_repeats: int = 20, cv_repeats: int = 3, bootstrap_resamples: int = 1000, external_path: Path | None = None, historical_holdout: bool = True) -> dict:
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, train_ids, test_ids, folds = split_dataset(dataset, test_size)
    if cv_repeats < 1:
        raise ValueError("cv_repeats must be positive.")
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=cv_repeats, random_state=SEED)
    splits = list(cv.split(X_train, y_train))
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "precision": make_scorer(precision_score, zero_division=0), "recall": make_scorer(recall_score, zero_division=0), "f1": make_scorer(f1_score, zero_division=0), "specificity": make_scorer(specificity)}
    models, unavailable = build_models()
    rows, fitted, fold_scores = [], {}, {}
    for name, pipeline in models.items():
        LOG.info("Training-only %d × %d-fold CV: %s", cv_repeats, folds, name)
        try:
            scores = cross_validate(pipeline, X_train, y_train, cv=splits, scoring=scoring, n_jobs=1, error_score="raise")
            row = {"model": name, "status": "trained", "cv_folds": folds, "cv_repeats": cv_repeats, "reason": "", "probability_method": "raw"}
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
    rows.sort(key=lambda row: (-row["cv_roc_auc_mean"], -row["cv_recall_mean"], -row["cv_precision_mean"], row["model"]))
    selected = rows[0]["model"]
    parameters = {name: {key: ("NaN (default missing marker)" if isinstance(value, float) and not np.isfinite(value) else value) for key, value in fitted[name].named_steps["classifier"].get_params().items()} for name in fitted}
    LOG.info("Base classifier selected from repeated training CV: %s. Comparing probability methods inside training only.", selected)
    calibration, final_pipeline = compare_calibration(models[selected], X_train, y_train, splits, folds=folds, repeats=cv_repeats)
    fitted[selected] = final_pipeline
    LOG.info("Probability method frozen: %s. Evaluating historical audit holdout.", calibration["selected_method"])
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
        row["test_role"] = "historical audit; previously inspected" if historical_holdout else "held-out descriptive evaluation"
    for name, reason in unavailable.items():
        LOG.warning("%s unavailable: %s", name, reason)
        rows.append({"model": name, "status": "unavailable", "reason": reason, "selected": False})
    comparison = pd.DataFrame(rows)
    run_id = uuid.uuid4().hex
    configuration = {"seed": SEED, "test_fraction": test_size, "cv_folds": folds, "cv_repeats": cv_repeats, "cv_total_splits": len(splits), "cv_shuffle": True, "cv_std_ddof": 0, "threshold": 0.5, "threshold_tuned": False, "hyperparameter_search": False, "synthetic_oversampling": False, "train_rows": len(X_train), "test_rows": len(X_test), "historical_holdout": historical_holdout, "bootstrap_resamples": bootstrap_resamples, "selection": "Highest training repeated-CV mean ROC-AUC for the base classifier; ties: recall, precision, then name. Probability method selected separately with nested training-only CV. Neither holdout nor external results select models or calibrators."}
    report = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "selected_model": selected, "data": dataset.metadata, "configuration": configuration, "cv_results": comparison[[c for c in comparison if not c.startswith("test_")]].where(pd.notna(comparison), None).to_dict(orient="records"), "cv_fold_scores": fold_scores, "test_results": test_metrics, "unavailable": unavailable, "model_parameters": parameters, "partition_indices": {"train": train_ids.tolist(), "test": test_ids.tolist()}, "class_counts": {"train": {str(k): int(v) for k, v in y_train.value_counts().items()}, "test": {str(k): int(v) for k, v in y_test.value_counts().items()}}, "limitations": ["Small, historical observational dataset; no external or prospective validation.", "Test performance is uncertain with so few observations; fold SD is not a confidence interval.", "Probabilities are uncalibrated model outputs, not clinical risk scores.", "Secondary test comparisons and held-out importance are descriptive; do not tune using them.", "Source selection, missingness, limited demographic encoding and hospital sampling can bias results."]}
    # Convert unavailable NaN cells to real JSON nulls, never non-standard NaN.
    report["cv_results"] = comparison[[c for c in comparison if not c.startswith("test_")]].astype(object).where(pd.notna(comparison), None).to_dict(orient="records")
    report["calibration"] = calibration
    selected_probability = probabilities[selected]
    selected_predictions = fitted[selected].predict(X_test)
    report["uncertainty"] = uncertainty(y_test, selected_predictions, selected_probability, resamples=bootstrap_resamples)
    report["subgroups"] = subgroup_results(X_test, y_test, selected_predictions, selected_probability)
    report["probability_scores"] = probability_scores(y_test, selected_probability)
    report["holdout_reliability"] = reliability_bins(y_test, selected_probability)
    report["test_interpretation"] = "Previously inspected Cleveland holdout retained as an audit. Updated results are not a fresh unbiased final benchmark." if historical_holdout else "Descriptive holdout evaluation; no clinical claims."
    report["external"] = {"status": "unavailable", "reason": "No external file supplied. Run scripts/download_external.py and train with --external data/raw/processed.hungarian.data."}
    if external_path is not None:
        from .external import evaluate_external
        LOG.info("Evaluating frozen pipeline on explicit external cohort: %s", external_path)
        try:
            report["external"] = evaluate_external(fitted[selected], dataset, external_path, resamples=bootstrap_resamples)
        except (ValueError, OSError) as exc:
            report["external"] = {"status": "unavailable", "reason": f"External validation/audit failed: {type(exc).__name__}: {exc}"}
            LOG.warning(report["external"]["reason"])
    report["limitations"] = ["Small historical observational cohorts; no prospective or clinical validation.", "Cleveland holdout has been inspected previously; new results are an audit, not a new final benchmark.", "Repeated folds are correlated; fold SD is not a confidence interval.", "Intervals condition on a fixed model and do not include training/selection or site uncertainty.", "Calibration is selected on training folds only; probability outputs are not clinically validated risk scores.", "Demographic slices are small and descriptive; no fairness or equivalence claim.", "External feature missingness and hospital selection can materially bias transfer results.", "Do not tune using viewed holdout, subgroup, importance or external diagnostics."]
    comparison.insert(0, "run_id", run_id)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    write_json(report_dir / "evaluation_report.json", report)
    write_json(report_dir / "data_provenance.json", dataset.metadata)
    write_json(report_dir / "calibration_report.json", {"run_id": run_id, **calibration})
    write_json(report_dir / "external_evaluation.json", {"run_id": run_id, **report["external"]})
    pd.DataFrame(report["subgroups"]).to_csv(report_dir / "subgroup_evaluation.csv", index=False)
    pd.DataFrame(calibration["fold_scores"]).to_csv(report_dir / "calibration_comparison.csv", index=False)
    # Each repeat average is shown separately; repeats are not treated as
    # independent observations for a confidence interval or significance test.
    repeat_rows = []
    for name, metrics in fold_scores.items():
        for repeat in range(cv_repeats):
            repeat_rows.append({"model": name, "repeat": repeat + 1, **{metric: float(np.mean(values[repeat * folds:(repeat + 1) * folds])) for metric, values in metrics.items()}})
    pd.DataFrame(repeat_rows).to_csv(report_dir / "cv_repeat_summary.csv", index=False)
    save_uncertainty_plot(report["uncertainty"], report_dir / "figures/metric_intervals.png")
    save_reliability_plot(calibration["reliability"], report_dir / "figures/training_reliability.png", title="Training out-of-fold probability reliability")
    save_reliability_plot({"Cleveland audit": report["holdout_reliability"]}, report_dir / "figures/holdout_reliability.png", title="Previously inspected Cleveland audit holdout")
    if report["external"]["status"] == "evaluated":
        pd.DataFrame(report["external"]["subgroups"]).to_csv(report_dir / "external_subgroups.csv", index=False)
        save_reliability_plot({"Hungarian external cohort": report["external"]["reliability"]}, report_dir / "figures/external_reliability.png", title="Frozen pipeline on the separate Hungarian site")
    save_diagnostic_plots(test_metrics, probabilities, y_test, selected, report_dir / "figures")
    save_cv_plot(comparison, report_dir / "figures/cv_comparison.png")
    importance = explain_model(fitted[selected], X_test, y_test, report_dir, repeats=importance_repeats)
    write_json(report_dir / "explainability.json", {"run_id": run_id, "method": "Original-feature permutation importance on final test set; ROC-AUC decrease", "repeats": importance_repeats, "seed": SEED, "test_rows": len(X_test), "features": importance.to_dict(orient="records"), "caution": "Descriptive only; correlated features share importance. Negative importance can occur. Repeat SD is not sampling uncertainty. No causal or clinical interpretation."})
    metadata = {"run_id": run_id, "created_utc": report["created_utc"], "selected_model": selected, "probability_method": calibration["selected_method"], "data": dataset.metadata, "training_configuration": configuration, "model_parameters": parameters[selected], "dependency_versions": dependency_versions(), "python_version": sys.version, "platform": platform.platform(), "evaluation": test_metrics[selected], "input_schema": {"numeric_bounds": BOUNDS, "categorical_codes": {col: list(codes) for col, codes in CATEGORY_LABELS.items()}, "missing_prediction_inputs": "rejected"}, "report_dir": str(report_dir)}
    save_artifact(fitted[selected], metadata, model_path)
    _write_project_report(report, comparison, importance, report_dir)
    _write_model_card(report, report_dir)
    return report


def _write_project_report(report: dict, comparison: pd.DataFrame, importance: pd.DataFrame, output_dir: Path) -> None:
    chosen = report["selected_model"]
    metrics = report["test_results"][chosen]
    config, data = report["configuration"], report["data"]
    table = ["| Model | CV AUC ± SD | CV recall | CV precision | Test AUC | Test accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in comparison.iterrows():
        if row["status"] == "trained":
            table.append(f"| {row['model']} | {row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f} | {row['cv_recall_mean']:.4f} | {row['cv_precision_mean']:.4f} | {row['test_roc_auc']:.4f} | {row['test_accuracy']:.4f} |")
        else:
            table.append(f"| {row['model']} | Unavailable: {row['reason']} | — | — | — | — |")
    text = f"""# Heart Disease Lab · experiment report

Run `{report['run_id']}` · {report['created_utc']}

## Data and quality

Source: {data['source']['source']}. {data['reference_url']}
Variant: {data['variant']}. SHA-256: `{data['sha256']}`.
Input: {data['raw_rows']} rows × {data['raw_columns']} columns. Used: {data['rows']} rows, {data['feature_count']} features.
Target: {data['target_mapping']}; counts {data['target_distribution']}.
Missing cells: {sum(data['missing_values'].values())}; exact duplicates removed: {data['duplicates_removed']}; corrected records: {data['records_corrected']}.
Quality notes: {'; '.join(data['notes']) or 'No records removed or corrected.'}

## Method

Stratified {config['train_rows']}/{config['test_rows']} train/test split; seed {config['seed']}.
{config['cv_repeats']} repeats of {config['cv_folds']}-fold stratified shuffled CV only within training. Every fold fits its own median imputation, numeric scaling for Logistic Regression, modal categorical imputation and one-hot encoding.
Five continuous measurements are numeric; eight coded variables are categorical (including vessel count, to avoid an assumed linear effect). No feature selection, tuning, oversampling or engineered clinical scores.
{config['selection']} Threshold fixed at 0.5. No refit on the final test data.
Full hyperparameters, fold scores and partition indices are in evaluation_report.json.

## Results

{chr(10).join(table)}

Selected before this audit evaluation: **{chosen}**. {report['test_interpretation']} Other test results are secondary descriptions of fixed competitors; do not use them to reselect models.
Selected historical audit: accuracy {metrics['accuracy']:.4f}, precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}, F1 {metrics['f1']:.4f}, specificity {metrics['specificity']:.4f}, ROC-AUC {metrics['roc_auc']:.4f}.
Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}.
False positives ({metrics['fp']}) are absence-labelled records predicted positive; false negatives ({metrics['fn']}) are presence-labelled records predicted negative. Neither output confirms a diagnosis.
Class-wise precision/recall/F1 and support are in evaluation_report.json.
Accuracy alone hides these different errors and is sensitive to class prevalence. Zero-denominator precision/recall/F1 is reported as 0; AUC is null if only one class exists (this experiment requires both).

## Explainability

Original-feature permutation importance uses {config['test_rows']} held-out records and ROC-AUC decrease. Highest computed means: {', '.join(importance.head(3)['feature'])}.
See permutation_importance.csv and figures/permutation_importance.png. This is descriptive and must not be used to tune this experiment. Repeat SD measures shuffle variability, not population uncertainty. Correlated inputs can share importance; negative values can occur.
transformed_model_weights.csv contains supplementary fitted coefficients (log odds per standardized numeric unit or encoded category) or native tree importance; these are not original-feature causal effects.

## Uncertainty, probabilities, subgroups and transfer

See model_card.md for actual interval and external results. evaluation_report.json includes Wilson 95% intervals for accuracy, precision, recall and specificity, plus stratified bootstrap ROC-AUC intervals ({config['bootstrap_resamples']} resamples). These condition on this fitted model and cohort; they exclude training/selection uncertainty.

Probability method: **{report['calibration']['selected_method']}**. {report['calibration']['selection_rule']}
calibration_comparison.csv contains paired outer-fold Brier/log-loss scores. training_reliability.png uses mean per-observation repeated out-of-fold predictions. {report['calibration']['caution']}

subgroup_evaluation.csv and external_subgroups.csv (when available) show predefined sex and age slices with counts and Wilson intervals. Empty denominators are null. Small slices cannot establish fairness or equivalence.

External cohort status: **{report['external']['status']}**. The external_evaluation.json audit records source hash, exclusions, missingness, range shifts, probabilities, metrics, intervals and whether any fitting occurred (always false). All external results are computed only after classifier and probability method are frozen. Do not tune against viewed transfer results.

cv_repeat_summary.csv shows each repeat separately. Repeated folds overlap; their SDs are not confidence intervals and do not imply a significant model difference.

## Limitations and responsible use

{chr(10).join('- ' + item for item in report['limitations'])}
- Associations and importance do not prove causation or clinical importance.
- No sex/age subgroup validation or fairness guarantees; the source has limited sex codes and historical hospital selection.
- No clinically calibrated probability of current disease or future disease incidence.
- Educational only. Do not use this project for medical decisions.

## Reproduce

Install requirements-lock.txt and the editable package, obtain the same SHA-256 source, then run scripts/eda.py and scripts/train.py as described in README.md. Saved metadata records package versions and exact configuration. Same-environment reruns are deterministic; other platforms/versions may differ slightly.
"""
    (output_dir / "project_report.md").write_text(text, encoding="utf-8")


def _write_model_card(report: dict, output_dir: Path):
    metrics = report["test_results"][report["selected_model"]]
    intervals = report["uncertainty"]["metrics"]
    table = ["| Metric | Estimate | 95% interval | Method |", "|---|---:|---:|---|"]
    for name, interval in intervals.items():
        if interval["estimate"] is None:
            table.append(f"| {name} | Undefined | — | {interval['method']} |")
        else:
            table.append(f"| {name} | {interval['estimate']:.4f} | {interval['lower']:.4f}–{interval['upper']:.4f} | {interval['method']} |")
    external = report["external"]
    if external["status"] == "evaluated":
        em = external["metrics"]
        external_text = f"{external['cohort']}: {external['data']['raw_rows']} source rows; {external['rows_evaluated']} evaluated after explicit deduplication/overlap auditing. Accuracy {em['accuracy']:.4f}, recall {em['recall']:.4f}, specificity {em['specificity']:.4f}, ROC-AUC {em['roc_auc']:.4f}. Source SHA-256 `{external['data']['sha256']}`. No fitting on external data. Exclusions: {external['exclusions']}.\n\n{external['caution']}"
    else:
        external_text = "Unavailable: " + external.get("reason", "No external evidence produced.")
    calibration_table = "\n".join(f"| {row['method']} | {row['brier_mean']:.5f} | {row['log_loss_mean']:.5f} |" for row in report["calibration"]["summary"])
    text = f"""# Model card — Heart Disease Lab

Run `{report['run_id']}` · {report['created_utc']}

## Intended use and users

Student portfolio, machine learning education and reproducibility demonstrations on public historical data. Outputs describe dataset labels. Not for diagnosis, treatment, triage, eligibility decisions or clinical deployment. Not a forecast of future disease.

## Model and data

Base classifier: **{report['selected_model']}**. Probability method: **{report['calibration']['selected_method']}**. Positive label: original UCI diagnosis >0; threshold 0.5, not tuned.
Training source: {report['data']['variant']}; SHA-256 `{report['data']['sha256']}`. {report['configuration']['train_rows']} training observations, 13 source features. Median numeric imputation, modal categorical imputation, one-hot encoding and scaling for Logistic Regression. If sigmoid is selected, each inner calibration fold contains the entire preprocessing/model pipeline.

## Evaluation protocol

{report['configuration']['selection']}
{report['configuration']['cv_repeats']} × {report['configuration']['cv_folds']}-fold repeated stratified training CV. Classifier ranking describes raw base models; probability-method comparison uses paired outer folds and independently fitted inner calibrators. Fold SDs do not represent independent-sample confidence intervals. No hyperparameter or threshold search.

## Previously inspected Cleveland audit holdout

{report['test_interpretation']} {report['configuration']['test_rows']} observations. Confusion matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}. F1: {metrics['f1']:.4f}.

{chr(10).join(table)}

{report['uncertainty']['caution']}

## Probability quality

| Training outer-CV method | Mean Brier (lower better) | Mean log loss (lower better) |
|---|---:|---:|
{calibration_table}

Selected **{report['calibration']['selected_method']}** using the predeclared training-only rule. Held-out audit Brier: {report['probability_scores']['brier']:.5f}; log loss: {report['probability_scores']['log_loss']:.5f}. These aggregate scores include discrimination and uncertainty; they do not measure calibration alone. See reliability plots and calibration_report.json. Calibration does not establish medical validity.

## Separate-site check

{external_text}

## Demographic slices and limitations

Predeclared recorded-sex groups and ages under 50, 50–59, and 60+. CSVs contain sample/class counts, metrics, Wilson accuracy/sensitivity/specificity intervals, and small-group warnings. No subgroup-driven fitting, fairness guarantee, significance test or equivalence claim. The source's binary sex coding is limited.

{chr(10).join('- ' + item for item in report['limitations'])}

## Reproducibility, privacy and trust

All transformations fit on training folds only. Run snapshots preserve model, metadata, reports and a completion manifest. Active outputs are published only after a complete run. No user input history or external prediction API is used. Demo inputs are clearly labelled public source records. Load only trusted joblib artifacts; checksums detect damage, not malicious files.

Reproduce with the README commands and requirements-lock.txt. Raw/model binaries are ignored by Git. The external source, original UCI attribution (Janosi et al., 1989; CC BY 4.0), exclusions and hashes are recorded in generated reports.
"""
    (output_dir / "model_card.md").write_text(text, encoding="utf-8")
