"""Cloud entrypoint: fetch one pinned trusted release, never retrain at startup."""
import os
import runpy
import sys
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

import streamlit as st

PROJECT = Path(__file__).resolve().parents[1]
# Load the stdlib-only bootstrap without importing the heart_disease package
# before its artifact paths and pinned source tree have been selected.
spec = spec_from_file_location("release_bootstrap", PROJECT / "src/heart_disease/bundle.py")
bootstrap = module_from_spec(spec)
spec.loader.exec_module(bootstrap)

try:
    bundle = bootstrap.ensure_cloud_bundle(Path(__file__).with_name("release.json"), PROJECT / ".deployment-cache")
except Exception as exc:
    st.set_page_config(page_title="Heart Disease Lab · deployment status", layout="centered")
    st.error(f"The pinned educational release could not be verified or loaded: {exc}")
    st.info("The maintainer must publish the release asset matching deploy/release.json. No fake model or automatic retraining is used.")
    st.stop()

os.environ["HEART_MODEL_PATH"] = str(bundle / "models/heart_model.joblib")
os.environ["HEART_REPORT_DIR"] = str(bundle / "reports")
os.environ["HEART_PUBLIC_DEMO"] = "1"
sys.path.insert(0, str(bundle / "src"))
runpy.run_path(str(bundle / "app.py"), run_name="__main__")
