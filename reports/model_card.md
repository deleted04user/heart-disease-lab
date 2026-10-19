# Model card — Heart Disease Lab

Run `27586d03470741438f644f6067b70f22` · 2026-10-19T14:37:51.374437+00:00

## Intended use and users

Student portfolio, machine learning education and reproducibility demonstrations on public historical data. Outputs describe dataset labels. Not for diagnosis, treatment, triage, eligibility decisions or clinical deployment. Not a forecast of future disease.

## Model and data

Base classifier: **Random Forest**. Probability method: **raw**. Positive label: original UCI diagnosis >0; threshold 0.5, not tuned.
Training source: UCI original processed Cleveland (303 records verified); SHA-256 `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`. 242 training observations, 13 source features. Median numeric imputation, modal categorical imputation, one-hot encoding and scaling for Logistic Regression. If sigmoid is selected, each inner calibration fold contains the entire preprocessing/model pipeline.

## Evaluation protocol

Highest training repeated-CV mean ROC-AUC for the base classifier; ties: recall, precision, then name. Probability method selected separately with nested training-only CV. Neither holdout nor external results select models or calibrators.
3 × 5-fold repeated stratified training CV. Classifier ranking describes raw base models; probability-method comparison uses paired outer folds and independently fitted inner calibrators. Fold SDs do not represent independent-sample confidence intervals. No hyperparameter or threshold search.

## Previously inspected Cleveland audit holdout

Previously inspected Cleveland holdout retained as an audit. Updated results are not a fresh unbiased final benchmark. 61 observations. Confusion matrix [[TN,FP],[FN,TP]]: [[27, 6], [3, 25]]. F1: 0.8475.

| Metric | Estimate | 95% interval | Method |
|---|---:|---:|---|
| accuracy | 0.8525 | 0.7428–0.9204 | Wilson |
| precision | 0.8065 | 0.6372–0.9081 | Wilson |
| recall | 0.8929 | 0.7280–0.9629 | Wilson |
| specificity | 0.8182 | 0.6561–0.9139 | Wilson |
| roc_auc | 0.9567 | 0.9004–0.9946 | Stratified percentile bootstrap |

Conditional on this fixed fitted model and sampled cohort. Does not include training, selection, site, or measurement uncertainty. AUC bootstrap preserves observed class counts. Wilson precision/accuracy can change with prevalence. Intervals do not establish clinical safety.

## Probability quality

| Training outer-CV method | Mean Brier (lower better) | Mean log loss (lower better) |
|---|---:|---:|
| raw | 0.13347 | 0.41939 |
| sigmoid | 0.13271 | 0.41683 |

Selected **raw** using the predeclared training-only rule. Held-out audit Brier: 0.10041; log loss: 0.34227. These aggregate scores include discrimination and uncertainty; they do not measure calibration alone. See reliability plots and calibration_report.json. Calibration does not establish medical validity.

## Separate-site check

UCI processed Hungarian: 294 source rows; 293 evaluated after explicit deduplication/overlap auditing. Accuracy 0.7952, recall 0.4906, specificity 0.9679, ROC-AUC 0.8987. Source SHA-256 `d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796`. No fitting on external data. Exclusions: {'exact_duplicates': 1, 'feature_overlap_with_cleveland': 0, 'invalid_rows_silently_removed': 0}.

Historical site-transfer check, not prospective clinical validation. Severe feature missingness can make Cleveland-trained imputation unsuitable. Original UCI diagnosis semantics are asserted and validated; input measurement bounds are enforced. No invalid rows are silently corrected or skipped. Feature-overlap audit cannot rule out shared patient identities. Do not tune using this cohort after viewing results.

## Demographic slices and limitations

Predeclared recorded-sex groups and ages under 50, 50–59, and 60+. CSVs contain sample/class counts, metrics, Wilson accuracy/sensitivity/specificity intervals, and small-group warnings. No subgroup-driven fitting, fairness guarantee, significance test or equivalence claim. The source's binary sex coding is limited.

- Small historical observational cohorts; no prospective or clinical validation.
- Cleveland holdout has been inspected previously; new results are an audit, not a new final benchmark.
- Repeated folds are correlated; fold SD is not a confidence interval.
- Intervals condition on a fixed model and do not include training/selection or site uncertainty.
- Calibration is selected on training folds only; probability outputs are not clinically validated risk scores.
- Demographic slices are small and descriptive; no fairness or equivalence claim.
- External feature missingness and hospital selection can materially bias transfer results.
- Do not tune using viewed holdout, subgroup, importance or external diagnostics.

## Reproducibility, privacy and trust

All transformations fit on training folds only. Run snapshots preserve model, metadata, reports and a completion manifest. Active outputs are published only after a complete run. No user input history or external prediction API is used. Demo inputs are clearly labelled public source records. Load only trusted joblib artifacts; checksums detect damage, not malicious files.

Reproduce with the README commands and requirements-lock.txt. Raw/model binaries are ignored by Git. The external source, original UCI attribution (Janosi et al., 1989; CC BY 4.0), exclusions and hashes are recorded in generated reports.
