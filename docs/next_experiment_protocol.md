# Next experiment protocol — measurement availability

Protocol version: 1.0, declared on 2026-10-19 before fetching the reserved Long Beach VA file. This is an educational retrospective experiment design, not a prospective clinical trial registration.

## Evidence already viewed

The Cleveland training partition and its previously inspected holdout, and the Hungarian external evaluation, have already been viewed. Their results may motivate hypotheses but cannot serve as a new unbiased confirmation set. The observed Hungarian missingness and low recall motivate an availability study; they do not prove a single causal mechanism.

## Fixed training and candidate definitions

- Retain exactly the existing 242 Cleveland training indices from run `27586d03470741438f644f6067b70f22`, seed 42. No Cleveland audit or Hungarian observations enter any fit, calibration, threshold selection, or candidate ranking.
- Candidate A: unchanged 13-feature Random Forest reference with the original fixed settings.
- Candidate B: the same fixed Random Forest using the 10 features excluding `ca`, `thal`, and `slope`. These exclusions are declared from observed availability before reading the reserved cohort; they are not selected from its outcomes.
- Candidate C: the same 10-feature pipeline with explicit missing indicators for numeric variables and a separate missing category for categorical variables. No generated patient records or outcome-conditioned imputation.
- All preprocessing belongs inside every fold. No hyperparameter search or SMOTE. Keep the probability threshold at 0.5; no Hungarian-derived threshold adjustment.

## Selection and reporting

Use identical three-repeat, five-fold stratified CV splits on training only. Primary ranking: mean ROC-AUC; tie-break if differences are below 0.005: higher mean recall, then lower mean Brier score, then the simpler candidate. The 0.005 tolerance is a predeclared software selection rule, not a statistical significance boundary. Report paired per-fold differences, all candidate means, SDs and class-wise errors; folds overlap and their SD is not a confidence interval.

Compare raw versus sigmoid probabilities for the training-selected candidate through inner calibration folds, keeping the existing predeclared Brier/log-loss retention rule. Freeze code/configuration, model, feature set, calibration method and threshold with a Git commit and checksums before releasing the reserved cohort for evaluation.

## Reserved confirmation cohort and release conditions

Reserve the official UCI `processed.va.data` (Long Beach V.A. Medical Center; source documentation describes 200 observations) as raw bytes only. Record its SHA-256 and the protocol SHA-256 without inspecting sample values, target distribution, missingness, or model outputs. The reservation helper must refuse accidental loading/evaluation of the reserved bytes in ordinary training/analysis paths.

After the candidate is frozen, a separate explicitly authorized confirmation run may audit the schema and evaluate exactly once. Invalid source values must be counted and explained; predeclared input constraints cannot be relaxed after looking at performance. If the schema is incompatible, report that confirmation is unavailable rather than inventing corrections or selecting another cohort after seeing outcomes. Do not claim this historical cohort is clinically representative or prospectively collected for this experiment.

Report ROC-AUC, accuracy, precision, recall, F1, specificity, class counts, confusion matrix, fixed-model intervals, probability reliability and descriptive demographic slices. Never reselect the candidate or threshold using the confirmation result. A poor result is retained and reported. No new candidate is promoted to the public demo by this protocol document alone.
