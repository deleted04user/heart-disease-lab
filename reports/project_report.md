# Heart Disease Lab · experiment report

Run `27586d03470741438f644f6067b70f22` · 2026-10-19T14:37:51.374437+00:00

## Data and quality

Source: UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease
Variant: UCI original processed Cleveland (303 records verified). SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`.
Input: 303 rows × 14 columns. Used: 303 rows, 13 features.
Target: 0 → 0; 1,2,3,4 → 1; counts {'0': 164, '1': 139}.
Missing cells: 6; exact duplicates removed: 0; corrected records: 0.
Quality notes: No records removed or corrected.

## Method

Stratified 242/61 train/test split; seed 42.
3 repeats of 5-fold stratified shuffled CV only within training. Every fold fits its own median imputation, numeric scaling for Logistic Regression, modal categorical imputation and one-hot encoding.
Five continuous measurements are numeric; eight coded variables are categorical (including vessel count, to avoid an assumed linear effect). No feature selection, tuning, oversampling or engineered clinical scores.
Highest training repeated-CV mean ROC-AUC for the base classifier; ties: recall, precision, then name. Probability method selected separately with nested training-only CV. Neither holdout nor external results select models or calibrators. Threshold fixed at 0.5. No refit on the final test data.
Full hyperparameters, fold scores and partition indices are in evaluation_report.json.

## Results

| Model | CV AUC ± SD | CV recall | CV precision | Test AUC | Test accuracy |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8959 ± 0.0414 | 0.7740 | 0.8038 | 0.9567 | 0.8525 |
| Logistic Regression | 0.8896 ± 0.0439 | 0.7651 | 0.8353 | 0.9665 | 0.8852 |
| XGBoost | 0.8836 ± 0.0384 | 0.7772 | 0.7951 | 0.9535 | 0.8852 |

Selected before this audit evaluation: **Random Forest**. Previously inspected Cleveland holdout retained as an audit. Updated results are not a fresh unbiased final benchmark. Other test results are secondary descriptions of fixed competitors; do not use them to reselect models.
Selected historical audit: accuracy 0.8525, precision 0.8065, recall 0.8929, F1 0.8475, specificity 0.8182, ROC-AUC 0.9567.
Confusion matrix [[TN, FP], [FN, TP]]: [[27, 6], [3, 25]].
False positives (6) are absence-labelled records predicted positive; false negatives (3) are presence-labelled records predicted negative. Neither output confirms a diagnosis.
Class-wise precision/recall/F1 and support are in evaluation_report.json.
Accuracy alone hides these different errors and is sensitive to class prevalence. Zero-denominator precision/recall/F1 is reported as 0; AUC is null if only one class exists (this experiment requires both).

## Explainability

Original-feature permutation importance uses 61 held-out records and ROC-AUC decrease. Highest computed means: ca, cp, oldpeak.
See permutation_importance.csv and figures/permutation_importance.png. This is descriptive and must not be used to tune this experiment. Repeat SD measures shuffle variability, not population uncertainty. Correlated inputs can share importance; negative values can occur.
transformed_model_weights.csv contains supplementary fitted coefficients (log odds per standardized numeric unit or encoded category) or native tree importance; these are not original-feature causal effects.

## Uncertainty, probabilities, subgroups and transfer

See model_card.md for actual interval and external results. evaluation_report.json includes Wilson 95% intervals for accuracy, precision, recall and specificity, plus stratified bootstrap ROC-AUC intervals (1000 resamples). These condition on this fitted model and cohort; they exclude training/selection uncertainty.

Probability method: **raw**. Retain sigmoid only if paired outer-CV mean Brier improves by >=0.002 and mean log loss also decreases; otherwise retain raw. All fitting and selection use training only.
calibration_comparison.csv contains paired outer-fold Brier/log-loss scores. training_reliability.png uses mean per-observation repeated out-of-fold predictions. Brier and log loss measure overall probability quality, not calibration alone. Reliability curves use per-observation mean repeated out-of-fold probabilities; they describe CV ensembles, not the final fitted model. Bin intervals are descriptive and ignore training dependence. Folds/repeats are correlated; score SD is not a confidence interval. Isotonic omitted because the calibration sample is small. Calibration is not clinical validation.

subgroup_evaluation.csv and external_subgroups.csv (when available) show predefined sex and age slices with counts and Wilson intervals. Empty denominators are null. Small slices cannot establish fairness or equivalence.

External cohort status: **evaluated**. The external_evaluation.json audit records source hash, exclusions, missingness, range shifts, probabilities, metrics, intervals and whether any fitting occurred (always false). All external results are computed only after classifier and probability method are frozen. Do not tune against viewed transfer results.

cv_repeat_summary.csv shows each repeat separately. Repeated folds overlap; their SDs are not confidence intervals and do not imply a significant model difference.

## Limitations and responsible use

- Small historical observational cohorts; no prospective or clinical validation.
- Cleveland holdout has been inspected previously; new results are an audit, not a new final benchmark.
- Repeated folds are correlated; fold SD is not a confidence interval.
- Intervals condition on a fixed model and do not include training/selection or site uncertainty.
- Calibration is selected on training folds only; probability outputs are not clinically validated risk scores.
- Demographic slices are small and descriptive; no fairness or equivalence claim.
- External feature missingness and hospital selection can materially bias transfer results.
- Do not tune using viewed holdout, subgroup, importance or external diagnostics.
- Associations and importance do not prove causation or clinical importance.
- No sex/age subgroup validation or fairness guarantees; the source has limited sex codes and historical hospital selection.
- No clinically calibrated probability of current disease or future disease incidence.
- Educational only. Do not use this project for medical decisions.

## Reproduce

Install requirements-lock.txt and the editable package, obtain the same SHA-256 source, then run scripts/eda.py and scripts/train.py as described in README.md. Saved metadata records package versions and exact configuration. Same-environment reruns are deterministic; other platforms/versions may differ slightly.
