"""Descriptive EDA from actual loaded records; no fitting or predictive claims."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import CATEGORICAL, CATEGORY_LABELS, FEATURES, NUMERIC, REPORT_DIR, TARGET
from .data import Dataset
from .io import write_json


def run_eda(dataset: Dataset, output_dir: Path = REPORT_DIR) -> dict:
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame, meta = dataset.frame, dataset.metadata
    sns.set_theme(style="whitegrid", palette=["#92B4F4", "#2463EB"])

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=160)
        plt.close(fig)

    counts = frame[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["Absence (0)", "Presence (1)"], counts, color=["#92B4F4", "#2463EB"])
    for i, count in enumerate(counts):
        ax.text(i, count, f"{count} · {count / len(frame):.1%}", ha="center", va="bottom")
    ax.set(ylabel="Records", title="Dataset label distribution", ylim=(0, counts.max() * 1.18))
    save(fig, "target_distribution.png")

    missing = frame[FEATURES].isna().sum()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(missing.index, missing.to_numpy(dtype=int), color="#2463EB")
    ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="Missing cells", title="Missing values before fold-local imputation")
    save(fig, "missing_values.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, ax in zip(NUMERIC, axes.flat, strict=False):
        sns.histplot(data=frame, x=col, bins=18, ax=ax, color="#2463EB")
        ax.set_title(col)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Numerical feature distributions")
    save(fig, "numeric_distributions.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, ax in zip(NUMERIC, axes.flat, strict=False):
        sns.boxplot(data=frame, x=TARGET, y=col, ax=ax, color="#92B4F4")
        ax.set(xlabel="Dataset target", title=col)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Measurements by dataset label · associations only")
    save(fig, "numeric_by_target.png")

    correlation = frame[NUMERIC + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7, 5.3))
    sns.heatmap(correlation, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Pearson correlation · continuous features and binary target")
    save(fig, "correlations.png")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for col, ax in zip(CATEGORICAL, axes.flat, strict=True):
        sns.countplot(data=frame, x=col, hue=TARGET, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("Original UCI code")
        ax.legend(title="Target", fontsize=7)
    fig.suptitle("Categorical distributions by dataset label")
    save(fig, "categorical_distributions.png")

    stats = frame[FEATURES].describe().T
    stats.to_csv(output_dir / "descriptive_statistics.csv")
    association = correlation[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    summary = {
        "data": meta,
        "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "positive_fraction": float(frame[TARGET].mean()),
        "class_ratio_majority_to_minority": float(counts.max() / counts.min()),
        "numeric_target_correlations": {
            key: (float(value) if pd.notna(value) else None) for key, value in association.items()
        },
        "categorical_labels": CATEGORY_LABELS,
    }
    write_json(output_dir / "eda_summary.json", summary)
    write_json(output_dir / "data_provenance.json", meta)
    missing_text = ", ".join(f"{col}: {int(n)}" for col, n in missing.items() if n) or "None"
    ranges = "\n".join(
        f"| {col} | {frame[col].min():.1f} | {frame[col].median():.1f} | {frame[col].max():.1f} |"
        for col in NUMERIC
    )
    report = f"""# Exploratory data analysis

Generated directly from `{meta["path"]}`, SHA-256 `{meta["sha256"]}`.
Variant: {meta["variant"]}. Source: {meta["source"]["source"]} ({meta["reference_url"]}).

## Overview and quality

Raw shape: {meta["raw_rows"]} × {meta["raw_columns"]}. Normalized shape: {frame.shape[0]} × {frame.shape[1]} ({len(FEATURES)} predictors and one binary target).
Feature names: {", ".join(FEATURES)}. Numeric storage is float64 to permit missing values; target is int64. Eight coded features are treated categorically in modeling.
Missing cells by feature: {missing_text}. Missing targets are rejected, never imputed.
Exact duplicates removed: {meta["duplicates_removed"]}. No values were silently corrected. {"; ".join(meta["notes"])}
Original target transformation: {meta["target_mapping"]}.
Class 0: {int(counts[0])}; class 1: {int(counts[1])}. Positive fraction: {frame[TARGET].mean():.2%}; majority/minority ratio: {counts.max() / counts.min():.3f}.

## Numerical observations

| Feature | Minimum | Median | Maximum |
|---|---:|---:|---:|
{ranges}

Pearson correlations with target: {", ".join(f"{col} = {value:.3f}" if pd.notna(value) else f"{col} = undefined (constant values)" for col, value in association.items())}.
These describe this sample and do not establish causality. Outliers are retained when they satisfy the documented entry schema; imputation is deferred to training folds.

## Figures and detailed inspection

- figures/target_distribution.png: class balance.
- figures/missing_values.png: feature missingness.
- figures/numeric_distributions.png: continuous distributions.
- figures/numeric_by_target.png: class-conditioned measurements.
- figures/correlations.png: continuous-feature relationships (coded categories excluded).
- figures/categorical_distributions.png: all eight categorical features by class.
- descriptive_statistics.csv: counts, mean, SD and quartiles; coded-variable means have limited meaning.
- eda_summary.json: computed types, class counts, correlations, mappings and provenance.

The notebook displays sample records locally. Full raw or processed records are intentionally not copied into reports intended for Git.

## Limitations

This historical hospital cohort is small and selected; missing values and site differences can affect models. Limited sex encoding cannot represent all people. Age and sex composition may create demographic bias; no fairness or external validation is established. Correlations, source labels and model probabilities are not a medical diagnosis or a forecast of future disease. EDA of the full dataset is descriptive; fixed model settings and CV selection do not depend on test metrics.

Attribution: Janosi, Steinbrunn, Pfisterer & Detrano (1989), Heart Disease, UCI Machine Learning Repository, https://doi.org/10.24432/C52P4X (CC BY 4.0).
"""
    (output_dir / "eda_report.md").write_text(report, encoding="utf-8")
    return summary
