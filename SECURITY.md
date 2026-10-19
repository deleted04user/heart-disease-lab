# Security policy

Report a vulnerability through the repository's GitHub Security tab using
"Report a vulnerability" rather than including exploit details or private data
in a public issue. Include the affected revision, reproduction and impact.
This volunteer educational project does not promise a response SLA.

Never upload clinical measurements, credentials, private datasets or unreviewed
joblib files. A joblib artifact can execute Python code: load only artifacts
created by the trusted maintainer and matching the pinned release/metadata
checksums. Checksums establish integrity against a trusted reference; they do
not make arbitrary third-party model files safe.

The hosted/container entrypoint accepts only bundled public UCI examples.
Dependency vulnerability checks run in CI and Dependabot proposes dependency,
action and base-image updates. These checks do not constitute clinical approval
or a guarantee that the software is free of vulnerabilities.
