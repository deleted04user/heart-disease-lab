# Release and Streamlit deployment

The source repository is `deleted04user/heart-disease-lab`. The public app uses
the `streamlit-demo` branch and `deploy/streamlit_app.py`, with Python **3.13**.

## Portable release

`python scripts/build_release.py` creates
`dist/heart-disease-lab-1.2.1.zip` and writes the exact archive SHA-256 and fixed
GitHub release URL to `deploy/release.json`. The archive contains the unchanged
trusted fitted model, matching metadata, sanitized aggregate reports, code,
licenses and five complete anonymous public UCI examples. It excludes full raw
datasets, reserved cohort bytes, user-submitted information and workstation paths.

`python scripts/verify_release.py` extracts into a separate temporary location,
starts an isolated Python process, verifies package origin/checksums/versions,
opens all dashboard pages and exercises two bundled examples with personal
measurement fields disabled. No raw data or editable project install is required.

The cloud entrypoint fetches only the explicitly pinned GitHub release archive,
checks its SHA-256 before extraction or model loading, rejects unsafe ZIP paths
and symlinks, verifies file hashes, and configures portable artifact paths. It
does not retrain or download clinical inputs at startup. Only public study records
can be selected; the server also checks that submitted form values match them.

Because joblib can execute code, the release asset must be built and published
by the trusted maintainer. Checksums do not make arbitrary third-party models safe.

## Local privacy and cloud binding

The main branch retains `127.0.0.1` in `.streamlit/config.toml` for private local
use. `python scripts/prepare_cloud_branch.py` derives a separate `streamlit-demo`
branch with the server address set to `0.0.0.0` for the cloud service. This does
not change the current checkout or expose the desktop's local server. Deploy
only `deploy/streamlit_app.py`, which enables public-example-only mode.
Use `python scripts/prepare_cloud_branch.py --update` to refresh an existing
derived branch after a main-branch change. The script preserves branch history
and verifies that only the intended configuration differs from main.

## Publish in order

1. Run unit tests, the publication index audit and clean portable verification.
2. Push reviewed source/aggregate materials on `main`; run the test matrix and
   manual real-data integration workflow.
3. Publish tag/release `v1.2.1` with the verified ZIP, without adding model binaries
   to Git history. Run the manual portable-release verification workflow.
4. Create/push `streamlit-demo`. In Streamlit Community Cloud choose repository
   `deleted04user/heart-disease-lab`, branch `streamlit-demo`, entrypoint
   `deploy/streamlit_app.py`, and Python 3.13 in Advanced settings.
5. Verify the deployed app's health, all pages and public-example predictions.
   Do not describe the application as live until the platform returns a working URL.

Community Cloud requires an authenticated account and acceptance of its terms.
The user must complete any sign-in/terms or newly requested account-permission
step; it is not bypassed with stored GitHub CLI credentials. No paid plan or
external medical prediction API is needed by this code.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Rebuilding a later release

Change the version, rerun all checks, build a new archive, and commit its checksum
before publishing a new release asset. Never replace bytes behind the same pinned
release. Keep the versioned experiment snapshots locally. The separate reserved
cohort stays excluded from release assets and must not be used to tune this demo.
