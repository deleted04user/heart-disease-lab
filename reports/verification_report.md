# Execution verification

## Engineering release 1.2.0 — 2026-10-19

- Windows/Python 3.13: **88 passed** in 38.23 seconds. Combined statement/branch
  coverage: **86.57%**, with no exclusions; CI enforces a minimum 85%.
- Ruff lint and format checks pass across source, scripts, tests, deployment and
  app; mypy checks all 18 package files, including unannotated function bodies.
- The universal uv lock and generated exports pin resolved versions and hashes.
  `pip-audit` found no known vulnerabilities in the pinned development export.
- New tests verify training/report/prediction coherence, populated pages, public
  example controls, immutable model use during transfer analysis, cache
  integrity, archive rejection, local health retries and publication exclusions.
- Six visible third-party plotting deprecation warnings remain. They are not
  suppressed. No synthetic fixture results are used as experiment evidence.
- Portable 1.2.0 passed isolated Windows and clean Linux verification. The five
  example probabilities match exactly; the fitted run and model bytes are unchanged.
  Archive SHA-256: `74b4939fcf49f554a6d7520028f2c185c014874ca50177861e3e4f68a8b59940`.
- The initial GitHub test matrix, real-data integration and portable verification
  have completed successfully. Current engineering/container CI evidence is
  recorded separately in `engineering_verification.json` and the Actions tab.
- Public Streamlit hosting remains deferred at the maintainer's request.

## Earlier research-release validation

Verified on 2026-10-19 in Windows with Python 3.13.14. Runtime versions are in `requirements-lock.txt`; optional video tooling is declared in `requirements-demo.txt`.

- Full updated suite: **67 passed**, no failures or test warnings (6.41 seconds on the final full-suite run).
- Fresh public-source Linux checkout with Python 3.10.20: **67 passed** in 7.12 seconds using the declared full dependencies. This is a local Linux compatibility run, separate from the still-pending GitHub runner results.
- Python syntax and dependency checks passed. The original 45 tests remain, with additional interval, calibration, external-evaluation, demo/export, versioned-run and interaction coverage.
- Repeated training-only CV completed for Logistic Regression, Random Forest and XGBoost: three repeats of five stratified folds, all using identical training partitions.
- Random Forest was selected by the declared repeated-CV rule. Nested raw/sigmoid probability comparisons ran using training data only. The raw method was retained because sigmoid did not meet the predeclared minimum Brier improvement, despite a small aggregate improvement.
- Original Cleveland: 303 actual UCI records, 242 training / 61 previously inspected audit observations. No new unbiased final-benchmark claim is made.
- Separate official Hungarian source: 294 source rows, one explicit exact duplicate removal, 293 evaluated. No Cleveland feature overlap was found. No external data was used for fitting, probability-method selection or tuning. Severe missingness and low external sensitivity are disclosed in the model card and dashboard.
- Wilson/binomial intervals, seeded AUC bootstrap intervals, reliability diagrams and predefined sex/age subgroup summaries were generated from actual evaluations. Undefined subgroup denominators remain null.
- `scripts/verify_run.py --server-url http://127.0.0.1:8501` passed: source hash, disjoint complete partitions, saved-model metrics, CV winner, JSON/images, all populated pages, blank form rejection, real-record/API consistency, calibration coherence, public-example/reset controls, exports excluding input measurements, reproduced external metrics and server health.
- The original experiment is preserved. Complete timestamped experiment folders contain model, metadata, aggregate reports and a completion manifest. Model/report hashes were verified. Active outputs were published only after completion.
- The source and release are published at `https://github.com/deleted04user/heart-disease-lab`. The Windows/Linux Python 3.10/3.13 matrix, manual real-data integration and portable release workflows have been triggered. Their remote results remain pending while GitHub reports queued jobs; see the repository Actions tab rather than interpreting a trigger as a pass.
- Six actual application screenshots support a 26-second, 1280 × 816 captioned MP4 walkthrough in `docs/demo.mp4`. Video generation and FFmpeg frame decoding succeeded. The walkthrough is explicitly a screenshot-based presentation, not a simulated live recording.
- The portable 1.1.0 archive passed isolated verification on Windows and in a clean Linux environment with serving dependencies only. All pages loaded, public-example controls worked, and the five example probabilities matched across platforms. Archive SHA-256: `f4702bf1fac3f690cbf5e62fcc7d1b078d892bef2435429a57960acbf54b4fd7`.
- The deployed model remains unchanged. Training-only out-of-fold missingness stress results and the future-cohort reservation are documented separately; the reserved bytes have not been parsed or evaluated.
- A fresh Linux clone of the public `streamlit-demo` branch successfully ran `deploy/streamlit_app.py`, downloaded the actual published release, opened all five pages and predicted two bundled examples. This verified the cloud bootstrap with no local fitted model or raw source data in the clone. Public hosting is deferred at the maintainer's request; Streamlit account sign-in/Terms acceptance remains required before hosted deployment.

The notebook's EDA logic remains shared with the successfully executed Python EDA workflow; an interactive Jupyter session was not run. The external check is historical and limited by missing data. No deployment, prospective clinical validation, fairness guarantee or medically validated probability is claimed.

Streamlit AppTest can print a benign `missing ScriptRunContext` warning outside the app server; this is not a failed UI check. Artificial records appear only in isolated tests, never in the reported experiments or fallback predictions.
