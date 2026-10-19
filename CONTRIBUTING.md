# Contributing

Use Python 3.10 or newer and uv 0.13.0. Install the exact development environment:

```sh
uv sync --locked --extra dev
uv run ruff check src scripts app.py tests deploy
uv run ruff format --check src scripts app.py tests deploy
uv run mypy src
uv run pytest --cov --cov-report=term-missing:skip-covered
uv run pip-audit -r requirements-lock.txt --no-deps --disable-pip --strict
uv build
```

CI enforces these checks and a minimum 85% combined statement/branch coverage
across the application and package. Tests use artificial records only in
isolated temporary directories; they are never reported as experiment results.
The full integration test covers fold separation, model persistence, metrics,
EDA, transfer diagnostics, strict JSON and populated dashboard interaction.

Keep a feature or fix and its behavioral regression tests in one focused commit
or pull request. Explain the problem, resulting behavior and verification.
Review feedback and contributions must reflect the people who actually made
them. Do not rewrite existing history to imply a different development process.

`pyproject.toml` declares dependencies and `uv.lock` resolves Python/platform
markers and hashes. After an intentional dependency change, run:

```sh
uv lock
uv export --locked --no-emit-project --no-dev --no-hashes --output-file requirements.txt --quiet
uv export --locked --no-emit-project --no-dev --extra dev --output-file requirements-lock.txt --quiet
```

The exports are generated files. CI checks that they remain synchronized.
Dependency update PRs require review; upgrades affecting fitted model versions
also require compatible artifact verification or a separately documented
training run. Do not change an existing release asset behind its pinned hash.

Preserve the existing model and the untouched reserved cohort. Changes to
selection, thresholds or training require the declared experimental protocol;
external results already seen must not become tuning targets. See the model
card and `docs/next_experiment_protocol.md`.

For container-based development and serving, see `docs/containers.md`.
