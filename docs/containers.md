# Isolated testing and serving

Docker Engine or Docker Desktop must be running. No raw datasets, local models,
secrets, run history or user inputs enter the build context. Both targets run as
UID 10001. The development target installs the full, locked test environment;
the smaller runtime target installs only the exact serving dependencies needed
by the pinned trusted release.

```sh
docker build --target development -t heart-disease-tests .
docker run --rm heart-disease-tests
docker build --target runtime -t heart-disease-demo .
docker run --rm heart-disease-demo python scripts/verify_cloud.py
docker run --rm -p 127.0.0.1:8502:8501 heart-disease-demo
```

Open http://127.0.0.1:8502. The app loads the SHA-pinned public release and only
allows its bundled UCI examples. It requires outbound HTTPS on the first startup
to obtain that asset. No training occurs at startup. The desktop's existing
localhost app on port 8501 is independent of this container.

`docker compose up --build` additionally uses a read-only root filesystem,
temporary cache, dropped capabilities and no-new-privileges. No personal dataset
volumes are mounted. Stop it with `docker compose down`.

The Docker healthcheck probes Streamlit's native `/_stcore/health`: it proves
process health, not model readiness. `scripts/verify_cloud.py` separately loads
the real cloud entrypoint, verifies package/model origin, exercises every page
and compares public-example form predictions with the saved pipeline. CI builds
both targets, runs the test suite inside the development image, verifies the
runtime image and checks the serving process health.

The development image includes Git because publication-regression tests create
temporary Git indexes. These tests never modify the checkout's actual index.
