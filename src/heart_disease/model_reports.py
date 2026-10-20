"""Render experiment reports from frozen evaluation results."""

from pathlib import Path

import pandas as pd


def _write_project_report(
    report: dict, comparison: pd.DataFrame, importance: pd.DataFrame, output_dir: Path
) -> None:
    chosen = report["selected_model"]
    metrics = report["test_results"][chosen]
    config, data = report["configuration"], report["data"]
    table = [
        "| Model | CV AUC ± SD | CV recall | CV precision | Test AUC | Test accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        if row["status"] == "trained":
            table.append(
                f"| {row['model']} | {row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f} | {row['cv_recall_mean']:.4f} | {row['cv_precision_mean']:.4f} | {row['test_roc_auc']:.4f} | {row['test_accuracy']:.4f} |"
            )
        else:
            table.append(f"| {row['model']} | Unavailable: {row['reason']} | — | — | — | — |")
    text = f"""# Heart Disease Lab · experiment report

Run `{report["run_id"]}` · {report["created_utc"]}

## Data and quality

Source: {data["source"]["source"]}. {data["reference_url"]}
Variant: {data["variant"]}. SHA-256: `{data["sha256"]}`.
Input: {data["raw_rows"]} rows × {data["raw_columns"]} columns. Used: {data["rows"]} rows, {data["feature_count"]} features.
Target: {data["target_mapping"]}; counts {data["target_distribution"]}.
Missing cells: {sum(data["missing_values"].values())}; exact duplicates removed: {data["duplicates_removed"]}; corrected records: {data["records_corrected"]}.
Quality notes: {"; ".join(data["notes"]) or "No records removed or corrected."}

## Method

Stratified {config["train_rows"]}/{config["test_rows"]} train/test split; seed {config["seed"]}.
{config["cv_repeats"]} repeats of {config["cv_folds"]}-fold stratified shuffled CV only within training. Every fold fits its own median imputation, numeric scaling for Logistic Regression, modal categorical imputation and one-hot encoding.
Five continuous measurements are numeric; eight coded variables are categorical (including vessel count, to avoid an assumed linear effect). No feature selection, tuning, oversampling or engineered clinical scores.
{config["selection"]} Threshold fixed at 0.5. No refit on the final test data.
Full hyperparameters, fold scores and partition indices are in evaluation_report.json.

## Results

{chr(10).join(table)}

Selected before this audit evaluation: **{chosen}**. {report["test_interpretation"]} Other test results are secondary descriptions of fixed competitors; do not use them to reselect models.
Selected historical audit: accuracy {metrics["accuracy"]:.4f}, precision {metrics["precision"]:.4f}, recall {metrics["recall"]:.4f}, F1 {metrics["f1"]:.4f}, specificity {metrics["specificity"]:.4f}, ROC-AUC {metrics["roc_auc"]:.4f}.
Confusion matrix [[TN, FP], [FN, TP]]: {metrics["confusion_matrix"]}.
False positives ({metrics["fp"]}) are absence-labelled records predicted positive; false negatives ({metrics["fn"]}) are presence-labelled records predicted negative. Neither output confirms a diagnosis.
Class-wise precision/recall/F1 and support are in evaluation_report.json.
Accuracy alone hides these different errors and is sensitive to class prevalence. Zero-denominator precision/recall/F1 is reported as 0; AUC is null if only one class exists (this experiment requires both).

## Explainability

Original-feature permutation importance uses {config["test_rows"]} held-out records and ROC-AUC decrease. Highest computed means: {", ".join(importance.head(3)["feature"])}.
See permutation_importance.csv and figures/permutation_importance.png. This is descriptive and must not be used to tune this experiment. Repeat SD measures shuffle variability, not population uncertainty. Correlated inputs can share importance; negative values can occur.
transformed_model_weights.csv contains supplementary fitted coefficients (log odds per standardized numeric unit or encoded category) or native tree importance; these are not original-feature causal effects.

## Uncertainty, probabilities, subgroups and transfer

See model_card.md for actual interval and external results. evaluation_report.json includes Wilson 95% intervals for accuracy, precision, recall and specificity, plus stratified bootstrap ROC-AUC intervals ({config["bootstrap_resamples"]} resamples). These condition on this fitted model and cohort; they exclude training/selection uncertainty.

Probability method: **{report["calibration"]["selected_method"]}**. {report["calibration"]["selection_rule"]}
calibration_comparison.csv contains paired outer-fold Brier/log-loss scores. training_reliability.png uses mean per-observation repeated out-of-fold predictions. {report["calibration"]["caution"]}

subgroup_evaluation.csv and external_subgroups.csv (when available) show predefined sex and age slices with counts and Wilson intervals. Empty denominators are null. Small slices cannot establish fairness or equivalence.

External cohort status: **{report["external"]["status"]}**. The external_evaluation.json audit records source hash, exclusions, missingness, range shifts, probabilities, metrics, intervals and whether any fitting occurred (always false). All external results are computed only after classifier and probability method are frozen. Do not tune against viewed transfer results.

cv_repeat_summary.csv shows each repeat separately. Repeated folds overlap; their SDs are not confidence intervals and do not imply a significant model difference.

## Limitations and responsible use

{chr(10).join("- " + item for item in report["limitations"])}
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
            table.append(
                f"| {name} | {interval['estimate']:.4f} | {interval['lower']:.4f}–{interval['upper']:.4f} | {interval['method']} |"
            )
    external = report["external"]
    if external["status"] == "evaluated":
        em = external["metrics"]
        external_text = f"{external['cohort']}: {external['data']['raw_rows']} source rows; {external['rows_evaluated']} evaluated after explicit deduplication/overlap auditing. Accuracy {em['accuracy']:.4f}, recall {em['recall']:.4f}, specificity {em['specificity']:.4f}, ROC-AUC {em['roc_auc']:.4f}. Source SHA-256 `{external['data']['sha256']}`. No fitting on external data. Exclusions: {external['exclusions']}.\n\n{external['caution']}"
    else:
        external_text = "Unavailable: " + external.get("reason", "No external evidence produced.")
    calibration_table = "\n".join(
        f"| {row['method']} | {row['brier_mean']:.5f} | {row['log_loss_mean']:.5f} |"
        for row in report["calibration"]["summary"]
    )
    text = f"""# Model card — Heart Disease Lab

Run `{report["run_id"]}` · {report["created_utc"]}

## Intended use and users

Student portfolio, machine learning education and reproducibility demonstrations on public historical data. Outputs describe dataset labels. Not for diagnosis, treatment, triage, eligibility decisions or clinical deployment. Not a forecast of future disease.

## Model and data

Base classifier: **{report["selected_model"]}**. Probability method: **{report["calibration"]["selected_method"]}**. Positive label: original UCI diagnosis >0; threshold 0.5, not tuned.
Training source: {report["data"]["variant"]}; SHA-256 `{report["data"]["sha256"]}`. {report["configuration"]["train_rows"]} training observations, 13 source features. Median numeric imputation, modal categorical imputation, one-hot encoding and scaling for Logistic Regression. If sigmoid is selected, each inner calibration fold contains the entire preprocessing/model pipeline.

## Evaluation protocol

{report["configuration"]["selection"]}
{report["configuration"]["cv_repeats"]} × {report["configuration"]["cv_folds"]}-fold repeated stratified training CV. Classifier ranking describes raw base models; probability-method comparison uses paired outer folds and independently fitted inner calibrators. Fold SDs do not represent independent-sample confidence intervals. No hyperparameter or threshold search.

## Previously inspected Cleveland audit holdout

{report["test_interpretation"]} {report["configuration"]["test_rows"]} observations. Confusion matrix [[TN,FP],[FN,TP]]: {metrics["confusion_matrix"]}. F1: {metrics["f1"]:.4f}.

{chr(10).join(table)}

{report["uncertainty"]["caution"]}

## Probability quality

| Training outer-CV method | Mean Brier (lower better) | Mean log loss (lower better) |
|---|---:|---:|
{calibration_table}

Selected **{report["calibration"]["selected_method"]}** using the predeclared training-only rule. Held-out audit Brier: {report["probability_scores"]["brier"]:.5f}; log loss: {report["probability_scores"]["log_loss"]:.5f}. These aggregate scores include discrimination and uncertainty; they do not measure calibration alone. See reliability plots and calibration_report.json. Calibration does not establish medical validity.

## Separate-site check

{external_text}

## Demographic slices and limitations

Predeclared recorded-sex groups and ages under 50, 50–59, and 60+. CSVs contain sample/class counts, metrics, Wilson accuracy/sensitivity/specificity intervals, and small-group warnings. No subgroup-driven fitting, fairness guarantee, significance test or equivalence claim. The source's binary sex coding is limited.

{chr(10).join("- " + item for item in report["limitations"])}

## Reproducibility, privacy and trust

All transformations fit on training folds only. Run snapshots preserve model, metadata, reports and a completion manifest. Active outputs are published only after a complete run. No user input history or external prediction API is used. Demo inputs are clearly labelled public source records. Load only trusted joblib artifacts; checksums detect damage, not malicious files.

Reproduce with the README commands and requirements-lock.txt. Raw/model binaries are ignored by Git. The external source, original UCI attribution (Janosi et al., 1989; CC BY 4.0), exclusions and hashes are recorded in generated reports.
"""
    (output_dir / "model_card.md").write_text(text, encoding="utf-8")
