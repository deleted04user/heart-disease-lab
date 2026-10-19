# Heart Disease Lab 1.3.0

This release hardens experiment persistence and activation while preserving the
fitted Random Forest run `27586d03470741438f644f6067b70f22` and its predictions.

- JSON writes stage and flush complete documents before replacing the previous
  file. Unique temporary files and a replacement lock handle concurrent threads
  on Windows; failed writes preserve the previous document and clean staging.
- Activation verifies model, metadata, run identity and the exact report
  inventory before changing active files. Extra/missing/tampered reports,
  symbolic links and unsafe paths are rejected. New schema-2 manifests also
  record a metadata digest, and completed manifests cannot be rewritten.
- `scripts/verify_experiment.py` checks a snapshot offline without opening
  source records or deserializing models. Legacy manifests remain supported and
  explicitly report that no metadata digest was recorded.
- Manual real-data and portable-release workflows now use pinned action
  commits and reproducible dependency installs. Real-data CI exercises the new
  snapshot verifier after training.
- Windows: 109 tests pass, one Linux-only symlink test skips, and combined
  statement/branch coverage is 87.20%, with no exclusions. Six third-party
  plotting deprecation warnings remain visible.
- All six [GitHub checks](https://github.com/deleted04user/heart-disease-lab/actions/runs/39316346892)
  pass, including Windows/Linux on Python 3.10/3.13 and development/runtime
  containers. The [real-data workflow](https://github.com/deleted04user/heart-disease-lab/actions/runs/39316371449)
  and [published portable verification](https://github.com/deleted04user/heart-disease-lab/actions/runs/39316378151)
  also pass at the release commit.

The portable archive SHA-256 is
`6add83f97ebed04de3d650f8090671cd83166cbd9ce816895471a2ce6202a9ba`.
Earlier published assets remain unchanged. Public hosting remains deferred.

Checksums establish snapshot consistency, not authenticity. Individual JSON
replacement is atomic; activating several files is not a multi-file transaction.
Use one activation at a time. No model tuning, reserved-cohort inspection or
clinical validation was performed for this engineering release.

## Previous release: 1.2.1

This engineering release preserves the fitted Random Forest run
`27586d03470741438f644f6067b70f22`, its raw probabilities and fixed threshold.

- Universal uv lock, explicit dependency metadata and hashed pip exports.
- Ruff lint/format and mypy gates; 85% combined coverage required by CI.
- 88 passing Windows tests and 86.57% combined coverage without exclusions.
- Full isolated training/report/transfer/UI test and bootstrap-integrity tests.
- Pinned-dependency vulnerability auditing and weekly Dependabot updates.
- Idempotent JSON operational logging and a documented environment template.
- Non-root development/serving containers and separate process/model readiness checks.
- Portable archive verified on Windows and clean Linux with identical example predictions.

The archive SHA-256 is
`32988a313897a9b98c68ca2175cc53b0ecf6a5e640c2ba688f524f9657cac5ea`.
Existing 1.1.0 assets remain unchanged. Public hosting remains deferred.

Six third-party plotting deprecation warnings are recorded. Historical Hungarian
recall remains 49.1%; no clinical validation or diagnostic suitability is claimed.
Code is MIT; the public UCI examples retain CC BY 4.0 attribution.
