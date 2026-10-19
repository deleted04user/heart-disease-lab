"""Uncertainty, nested training-only probability checks, and subgroup summaries."""
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import SEED
from .evaluation import evaluate


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> dict:
    """Wilson score interval; undefined denominators are represented by nulls."""
    if total < 0 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence level.")
    if total == 0:
        return {"estimate": None, "lower": None, "upper": None, "n": 0, "method": "Wilson"}
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    half = z * np.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total)) / denominator
    return {"estimate": estimate, "lower": 0.0 if successes == 0 else max(0.0, float(center - half)), "upper": 1.0 if successes == total else min(1.0, float(center + half)), "n": total, "method": "Wilson"}


def uncertainty(y_true, predictions, probabilities, *, resamples: int = 1000, seed: int = SEED) -> dict:
    """Fixed-model intervals; AUC bootstrap preserves the observed class counts."""
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples.")
    y, predicted, probability = np.asarray(y_true), np.asarray(predictions), np.asarray(probabilities)
    if not len(y) or not (len(y) == len(predicted) == len(probability)):
        raise ValueError("Uncertainty inputs must be non-empty and aligned.")
    metrics = evaluate(y, predicted, probability)
    tp, tn, fp, fn = (metrics[key] for key in ["tp", "tn", "fp", "fn"])
    result = {
        "accuracy": wilson_interval(tp + tn, len(y)),
        "precision": wilson_interval(tp, tp + fp),
        "recall": wilson_interval(tp, tp + fn),
        "specificity": wilson_interval(tn, tn + fp),
    }
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if len(negative) and len(positive):
        rng = np.random.default_rng(seed)
        aucs = []
        for _ in range(resamples):
            indices = np.concatenate([rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)])
            aucs.append(roc_auc_score(y[indices], probability[indices]))
        lower, upper = np.quantile(aucs, [0.025, 0.975])
        result["roc_auc"] = {"estimate": metrics["roc_auc"], "lower": float(lower), "upper": float(upper), "n": len(y), "method": "Stratified percentile bootstrap", "resamples": resamples, "seed": seed}
    else:
        result["roc_auc"] = {"estimate": None, "lower": None, "upper": None, "n": len(y), "method": "Undefined: one class only"}
    return {"confidence": 0.95, "metrics": result, "caution": "Conditional on this fixed fitted model and sampled cohort. Does not include training, selection, site, or measurement uncertainty. AUC bootstrap preserves observed class counts. Wilson precision/accuracy can change with prevalence. Intervals do not establish clinical safety."}


def probability_scores(y, probabilities) -> dict:
    return {"brier": float(brier_score_loss(y, probabilities)), "log_loss": float(log_loss(y, probabilities, labels=[0, 1]))}


def make_probability_pipeline(base_pipeline, method: str, *, inner_folds: int = 3):
    if method == "raw":
        return clone(base_pipeline)
    if method != "sigmoid" or inner_folds < 2:
        raise ValueError("Supported probability methods: raw, sigmoid; >=2 inner folds required.")
    # The WHOLE pipeline is inside calibration: every inner fold learns its own
    # preprocessing, never preprocessing fitted before the calibration split.
    calibrated = CalibratedClassifierCV(
        estimator=clone(base_pipeline), method="sigmoid",
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED),
        ensemble=True, n_jobs=1,
    )
    return Pipeline([("classifier", calibrated)])


def reliability_bins(y, probabilities, *, bins: int = 5) -> list[dict]:
    """Fixed equal-width bins with binomial intervals; empty bins stay visible."""
    y, probability = np.asarray(y), np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1)
    assignment = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    output = []
    for index in range(bins):
        mask = assignment == index
        count = int(mask.sum())
        interval = wilson_interval(int(y[mask].sum()), count)
        output.append({"bin": index + 1, "lower_edge": float(edges[index]), "upper_edge": float(edges[index + 1]), "n": count, "mean_probability": float(probability[mask].mean()) if count else None, "positive_fraction": interval["estimate"], "lower": interval["lower"], "upper": interval["upper"]})
    return output


def compare_calibration(base_pipeline, X_train, y_train, splits: list, *, folds: int, repeats: int) -> tuple[dict, object]:
    """Outer validation compares raw and sigmoid; inner CV fits calibrators."""
    methods = ["raw", "sigmoid"]
    rows, sums, counts = [], {key: np.zeros(len(X_train)) for key in methods}, np.zeros(len(X_train))
    for index, (fit_ids, validation_ids) in enumerate(splits):
        inner_folds = min(3, int(y_train.iloc[fit_ids].value_counts().min()))
        if inner_folds < 2:
            # A small training dataset still gets a valid raw-probability check.
            methods = ["raw"]
        for method in methods:
            candidate = make_probability_pipeline(base_pipeline, method, inner_folds=max(2, inner_folds))
            candidate.fit(X_train.iloc[fit_ids], y_train.iloc[fit_ids])
            probability = candidate.predict_proba(X_train.iloc[validation_ids])[:, 1]
            scores = probability_scores(y_train.iloc[validation_ids], probability)
            rows.append({"method": method, "repeat": index // folds + 1, "fold": index % folds + 1, **scores})
            sums[method][validation_ids] += probability
        counts[validation_ids] += 1
    table = pd.DataFrame(rows)
    summary = table.groupby("method")[["brier", "log_loss"]].agg(["mean", "std"])
    raw_brier, raw_log_loss = summary.loc["raw", ("brier", "mean")], summary.loc["raw", ("log_loss", "mean")]
    chosen = "raw"
    if "sigmoid" in summary.index:
        sigmoid_brier, sigmoid_log_loss = summary.loc["sigmoid", ("brier", "mean")], summary.loc["sigmoid", ("log_loss", "mean")]
        if raw_brier - sigmoid_brier >= 0.002 and sigmoid_log_loss < raw_log_loss:
            chosen = "sigmoid"
    # One full out-of-fold probability per observation per repeat, with separate
    # reliability curves. No row appears repeatedly in a Wilson bin interval.
    curves = {}
    for method in methods:
        curves[method] = reliability_bins(y_train, sums[method] / counts)
    full_inner_folds = min(3, int(y_train.value_counts().min()))
    final = make_probability_pipeline(base_pipeline, chosen, inner_folds=full_inner_folds)
    final.fit(X_train, y_train)
    report = {
        "selected_method": chosen, "outer_folds": folds, "outer_repeats": repeats,
        "inner_folds_max": 3, "training_rows": len(X_train),
        "score_sd_ddof": 1,
        "selection_rule": "Retain sigmoid only if paired outer-CV mean Brier improves by >=0.002 and mean log loss also decreases; otherwise retain raw. All fitting and selection use training only.",
        "fold_scores": rows,
        "summary": [{"method": method, "brier_mean": float(summary.loc[method, ("brier", "mean")]), "brier_sd": float(summary.loc[method, ("brier", "std")]) if len(splits) > 1 else 0.0, "log_loss_mean": float(summary.loc[method, ("log_loss", "mean")]), "log_loss_sd": float(summary.loc[method, ("log_loss", "std")]) if len(splits) > 1 else 0.0} for method in methods],
        "reliability": curves,
        "caution": "Brier and log loss measure overall probability quality, not calibration alone. Reliability curves use per-observation mean repeated out-of-fold probabilities; they describe CV ensembles, not the final fitted model. Bin intervals are descriptive and ignore training dependence. Folds/repeats are correlated; score SD is not a confidence interval. Isotonic omitted because the calibration sample is small. Calibration is not clinical validation.",
    }
    return report, final


def save_reliability_plot(curves: dict, path, *, title: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, bins in curves.items():
        rows = [row for row in bins if row["n"]]
        x = np.array([row["mean_probability"] for row in rows])
        y = np.array([row["positive_fraction"] for row in rows])
        lower = y - np.array([row["lower"] for row in rows])
        upper = np.array([row["upper"] for row in rows]) - y
        ax.errorbar(x, y, yerr=[lower, upper], fmt="o-", capsize=3, label=name)
    ax.plot([0, 1], [0, 1], "--", color="#8492A6", label="Perfect reliability")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean model-estimated probability", ylabel="Observed positive fraction", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def subgroup_results(X, y, predictions, probabilities) -> list[dict]:
    """Predeclared demographic slices, no threshold changes or fairness claim."""
    y, predicted, probability = np.asarray(y), np.asarray(predictions), np.asarray(probabilities)
    groups = [("Recorded sex", "Female (0)", np.asarray(X.sex == 0)), ("Recorded sex", "Male (1)", np.asarray(X.sex == 1))]
    for label, low, high in [("Under 50", 0, 50), ("50–59", 50, 60), ("60 and older", 60, 121)]:
        groups.append(("Age", label, np.asarray((X.age >= low) & (X.age < high))))
    rows = []
    for attribute, label, mask in groups:
        n = int(mask.sum())
        if not n:
            rows.append({"attribute": attribute, "group": label, "n": 0, "positive_n": 0, "negative_n": 0, "accuracy": None, "recall": None, "specificity": None, "precision": None, "roc_auc": None, "accuracy_lower": None, "accuracy_upper": None, "recall_lower": None, "recall_upper": None, "specificity_lower": None, "specificity_upper": None, "caution": "No observations"})
            continue
        result = evaluate(y[mask], predicted[mask], probability[mask])
        row = {"attribute": attribute, "group": label, "n": n, "positive_n": int(y[mask].sum()), "negative_n": int((y[mask] == 0).sum())}
        for key in ["accuracy", "recall", "specificity", "precision", "roc_auc"]:
            row[key] = result[key]
        for key, successes, total in [("accuracy", result["tp"] + result["tn"], n), ("recall", result["tp"], result["tp"] + result["fn"]), ("specificity", result["tn"], result["tn"] + result["fp"])]:
            interval = wilson_interval(successes, total)
            row[key] = interval["estimate"]
            row[key + "_lower"], row[key + "_upper"] = interval["lower"], interval["upper"]
        if result["tp"] + result["fp"] == 0:
            row["precision"] = None
        row["caution"] = "Small or single-class subgroup; descriptive only" if min(row["positive_n"], row["negative_n"]) < 10 else "Descriptive; no fairness or equivalence claim"
        rows.append(row)
    return rows


def save_uncertainty_plot(intervals: dict, path):
    rows = [(name, value) for name, value in intervals["metrics"].items() if value["estimate"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    means = np.array([value["estimate"] for _, value in rows])
    lower = means - np.array([value["lower"] for _, value in rows])
    upper = np.array([value["upper"] for _, value in rows]) - means
    ax.errorbar(means, [name.replace("_", " ").title() for name, _ in rows], xerr=[lower, upper], fmt="o", color="#2463EB", capsize=4)
    ax.set(xlim=(0, 1.02), xlabel="Estimate with 95% interval", title="Cleveland audit holdout · fixed model uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
