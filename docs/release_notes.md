# Heart Disease Lab 1.2.1

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
