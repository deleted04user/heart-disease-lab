# Heart Disease Lab 1.2.0

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
`74b4939fcf49f554a6d7520028f2c185c014874ca50177861e3e4f68a8b59940`.
Existing 1.1.0 assets remain unchanged. Public hosting remains deferred.

Six third-party plotting deprecation warnings are recorded. Historical Hungarian
recall remains 49.1%; no clinical validation or diagnostic suitability is claimed.
Code is MIT; the public UCI examples retain CC BY 4.0 attribution.
