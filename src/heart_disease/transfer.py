"""Descriptive transfer diagnostics and training-only missingness stress checks."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .config import FEATURES, SEED
from .data import load_dataset
from .evaluation import evaluate
from .io import read_json, sha256, write_json
from .prediction import load_artifact
from .research import wilson_interval


def investigate(model_path: Path, report_dir: Path) -> dict:
    pipeline, metadata = load_artifact(model_path)
    evaluation = read_json(report_dir / "evaluation_report.json")
    if metadata["run_id"] != evaluation["run_id"]:
        raise ValueError("Use model and reports from the same run.")
    data = load_dataset(Path(metadata["data"]["path"]))
    if data.metadata["sha256"] != evaluation["data"]["sha256"]:
        raise ValueError("Original source changed since training.")
    external = evaluation.get("external", {})
    if external.get("status") != "evaluated":
        raise ValueError("The existing, already-viewed external evaluation is required.")
    path = Path(external["data"]["path"])
    if not path.is_absolute():
        from .config import ROOT
        path = ROOT / path
    cohort = load_dataset(path)
    if cohort.metadata["sha256"] != external["data"]["sha256"]:
        raise ValueError("External source changed since the reported evaluation.")
    overlap = external["overlapping_normalized_row_indices_excluded"]
    cohort_frame = cohort.frame.drop(index=overlap)
    train = data.frame.iloc[evaluation["partition_indices"]["train"]]
    X, y = train[FEATURES], train.target
    folds = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    groups = {"Unmodified validation records": [], "Mask ca and thal in validation": ["ca", "thal"], "Mask ca, thal and slope in validation": ["ca", "thal", "slope"]}
    outputs = {name: np.zeros(len(X)) for name in groups}
    for fit_ids, validation_ids in cv.split(X, y):
        fitted = clone(pipeline).fit(X.iloc[fit_ids], y.iloc[fit_ids])
        for name, columns in groups.items():
            validation = X.iloc[validation_ids].copy()
            if columns:
                validation[columns] = np.nan
            outputs[name][validation_ids] = fitted.predict_proba(validation)[:, 1]
    stress = [{"condition": name, "records": len(X), **{metric: value for metric, value in evaluate(y, probability >= 0.5, probability).items() if metric in {"accuracy", "recall", "specificity", "roc_auc"}}} for name, probability in outputs.items()]
    missingness = [{"feature": col, "cleveland_training_missing_fraction": float(X[col].isna().mean()), "hungarian_missing_fraction": float(cohort_frame[col].isna().mean())} for col in FEATURES]
    external_probability = pipeline.predict_proba(cohort_frame[FEATURES])[:, 1]
    strata = []
    for col in ["ca", "thal", "slope"]:
        for missing in [False, True]:
            mask = cohort_frame[col].isna().to_numpy() == missing
            positive = mask & (cohort_frame.target.to_numpy() == 1)
            positive_n = int(positive.sum())
            detected = int((external_probability[positive] >= 0.5).sum())
            interval = wilson_interval(detected, positive_n)
            strata.append({"feature": col, "state": "missing" if missing else "observed", "n": int(mask.sum()), "positive_n": positive_n, "true_positive_n": detected, "recall": interval["estimate"], "recall_lower": interval["lower"], "recall_upper": interval["upper"], "caution": "Observational slice; selection/confounding prevents causal attribution"})
    result = {
        "run_id": metadata["run_id"], "model_sha256": sha256(model_path),
        "training_source_sha256": data.metadata["sha256"], "external_source_sha256": cohort.metadata["sha256"],
        "fitting_rows": len(train), "holdout_rows_used_for_fitting_or_selection": 0,
        "external_rows_used_for_fitting_or_selection": 0,
        "training_only_stress": stress, "missingness": missingness, "observed_external_slices": strata,
        "interpretation": "Stress checks mask predictors on genuine training-fold validation observations. They are controlled perturbations, not a new natural cohort or fabricated dataset. Changes show dependence on feature availability, not proof that missingness alone caused the Hungarian recall gap. Site selection, class prevalence, covariate differences and missingness mechanisms remain confounded. Observed external slices may be tiny and must not select models or thresholds.",
        "deployment_model_changed": False,
    }
    write_json(report_dir / "transfer_diagnostics.json", result)
    pd.DataFrame(stress).to_csv(report_dir / "training_missingness_stress.csv", index=False)
    pd.DataFrame(missingness).to_csv(report_dir / "transfer_missingness.csv", index=False)
    pd.DataFrame(strata).to_csv(report_dir / "transfer_observed_slices.csv", index=False)
    lines = "\n".join(f"| {row['condition']} | {row['recall']:.4f} | {row['specificity']:.4f} | {row['roc_auc']:.4f} |" for row in stress)
    missing_lines = "\n".join(f"| {row['feature']} | {row['cleveland_training_missing_fraction']:.1%} | {row['hungarian_missing_fraction']:.1%} |" for row in sorted(missingness, key=lambda row: -row["hungarian_missing_fraction"]))
    text = f"""# Why does recall fall on the Hungarian cohort?

Descriptive investigation of frozen run `{metadata['run_id']}`. No model, probability method or threshold was changed. No future reserved cohort was read.

The existing Hungarian check reports recall {external['metrics']['recall']:.4f}, specificity {external['metrics']['specificity']:.4f}, and ROC-AUC {external['metrics']['roc_auc']:.4f}. At the fixed 0.5 threshold the model misses {external['metrics']['fn']} of {external['metrics']['fn'] + external['metrics']['tp']} presence-labelled observations. Ranking performance and threshold sensitivity describe different properties.

## Observed availability shift

| Feature | Cleveland training missing | Hungarian missing |
|---|---:|---:|
{missing_lines}

The leading importance features include vessel count and chest-pain type. Missing vessel/thallium/slope measurements are replaced by Cleveland-trained imputation values. A site with a different availability pattern can therefore be represented poorly even when its outcome ranking appears useful.

## Controlled training-only stress check

{folds}-fold stratified out-of-fold predictions on the {len(train)} original training observations, seed 42. Every fit uses only that fold's training subset. The fitted model is unchanged between a fold's validation conditions. Only validation feature availability is deliberately perturbed.

| Validation condition | Recall | Specificity | ROC-AUC |
|---|---:|---:|---:|
{lines}

{result['interpretation']}

External missing/observed slices with sample counts and Wilson recall intervals are in transfer_observed_slices.csv. Their class composition and small observed groups prevent a simple causal comparison. No attempt was made to fill missing Hungarian measurements with invented observations or to tune against its already-viewed outcomes.

## Next step

See docs/next_experiment_protocol.md. Future candidates and selection criteria are declared using training data and observed measurement availability. A different UCI cohort is reserved without opening its labels or evaluating a model. The current deployed model remains the verified original run.
"""
    (report_dir / "transfer_diagnostics.md").write_text(text, encoding="utf-8")
    return result
