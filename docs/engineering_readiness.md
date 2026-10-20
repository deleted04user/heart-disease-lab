# Engineering readiness evidence

This maps the requested engineering checks to executable repository evidence.
It describes implementation and verification, rather than predicting an
external quality grade or commercial value.

| Item | Implementation and evidence | Status |
| --- | --- | --- |
| 1. Pair changes with regression tests | Focused commits include both behavior and tests: `965198f` (experiment verification), `b430d87` (input conversion), `7e3accb` (release cache). | Implemented; continue for future work |
| 2. Install, build and test from a clean clone | README's locked uv commands; CI checks fresh Linux/Windows environments on Python 3.10/3.13, builds distributions and exercises both container targets. | Verified in CI |
| 3. Sustained development | Baseline `131f84f` has 22 commits and one author over less than one day. | Requires real development over time |
| 4. Committed, pinned dependencies | `pyproject.toml`, `uv.lock`, hashed `requirements-lock.txt` and `deploy/requirements-lock.txt`; CI checks lock exports and audits both dependency graphs. | Implemented and enforced |
| 5. Lint and type checks | Ruff lint/format and mypy are required in `lint-and-typecheck`; pytest requires 85% combined statement/branch coverage with no exclusions. | Implemented and enforced |
| 6. Incremental, test-backed history | Existing focused feature/fix commits preserve their regression tests; contributions keep their actual author identities. | Implemented practice; maintenance depth remains limited |
| 7. Dependency vulnerability scanning | Both real locks are audited. `scripts/verify_audit_gate.py` additionally requires a known vulnerable fixture to return a failing auditor exit and the expected advisory. | Implemented with a rejection check |
| 8. Structured logging | `src/heart_disease/logging_config.py`, `tests/test_logging.py`, `HEART_LOG_LEVEL` and `.env.example`; CLI/app entrypoints configure JSON logging. | Implemented and tested |
| 9. Self-contained containers | `Dockerfile` development/runtime targets, `compose.yaml`, `docs/containers.md`, native process-health and model/UI readiness checks in CI. | Implemented and verified |

The [1.3.1 source checks](https://github.com/deleted04user/heart-disease-lab/actions/runs/39334926949)
cover the existing matrix, containers, lock checks and audits. Current workflow
runs remain available from [GitHub Actions](https://github.com/deleted04user/heart-disease-lab/actions).
`reports/input_cache_verification.json` retains the earlier verification record.

## Check the audit's failure path

```sh
uv run --locked python scripts/verify_audit_gate.py
```

The probe writes `urllib3==1.26.4` to a temporary requirements file and queries
the advisory service using the same strict, no-resolution audit flags as CI.
It never installs that version or changes the real lockfiles. The
[upstream advisory](https://github.com/urllib3/urllib3/security/advisories/GHSA-q2q7-5pp4-w6pg)
documents the historical affected range. The verifier checks both exit status
and advisory IDs; a network error, skipped dependency, empty result or unrelated
finding fails the check. Offline tests exercise those outcomes.

The normal audits still require clean results for the real locks. The sentinel
does not suppress findings, replace those audits or establish that code has no
unknown vulnerabilities. Advisory-service unavailability fails CI rather than
being treated as successful detection.

Existing published releases and the fitted model are unchanged by this CI-only
verification addition. No app reload or new local model training is required.
