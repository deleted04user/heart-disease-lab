"""Extract a pinned bundle, then verify it from outside the original source tree."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import ROOT
from heart_disease.bundle import checksum, extract_bundle

CHILD = r"""
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
os.environ["HEART_PUBLIC_DEMO"] = "1"
os.environ["HEART_MODEL_PATH"] = str(root / "models/heart_model.joblib")
os.environ["HEART_REPORT_DIR"] = str(root / "reports")
from heart_disease import prediction
from heart_disease.demo import public_demo_record
from heart_disease.prediction import load_artifact, predict_records
assert Path(prediction.__file__).resolve().is_relative_to(root), "Original checkout shadowed portable package"
pipeline, metadata = load_artifact(root / "models/heart_model.joblib")
assert not (root / "data/raw/processed.cleveland.data").exists()
records = [public_demo_record(metadata, index)[0] for index in range(metadata["public_demo"]["count"])]
predictions = predict_records(pipeline, records)
from streamlit.testing.v1 import AppTest
app = AppTest.from_file(str(root / "app.py")).run(timeout=30)
assert not app.exception
for page in ["Model Evaluation", "Explainability", "About & Limitations", "Patient Input & Prediction"]:
    app.sidebar.radio[0].set_value(page).run(timeout=30)
    assert not app.exception, page
assert all(widget.disabled for widget in app.number_input)
app.button(key="load_demo").click().run(timeout=30)
assert not app.exception and not app.error
app.button(key="predict_submit").click().run(timeout=30)
assert not app.exception and not app.error
expected = predictions.iloc[0].positive_class_probability
assert any(metric.value == f"{expected:.1%}" for metric in app.metric)
app.selectbox(key="demo_example_index").set_value(len(records)-1).run(timeout=30)
app.button(key="load_demo").click().run(timeout=30)
app.button(key="predict_submit").click().run(timeout=30)
assert not app.exception and not app.error
assert app.session_state["prediction_export"]["public_example_loaded"] is True
import platform
print(json.dumps({"status":"passed", "os":platform.system(), "python":platform.python_version(), "run_id":metadata["run_id"], "examples":len(records), "probabilities":predictions.positive_class_probability.tolist(), "checks":["portable package origin", "no raw dataset required", "artifact checksum and versions", "all dashboard pages", "public-only input controls", "two public examples predicted", "form/API consistency"]}))
"""


def verify(bundle: Path, output: Path | None = None):
    payload = bundle.read_bytes()
    with tempfile.TemporaryDirectory(prefix="heart-portable-") as directory:
        extracted = Path(directory) / "app"
        extract_bundle(payload, extracted, checksum(payload))
        result = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, str(extracted)],
            cwd=directory,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        lines = result.stdout.strip().splitlines()
        report = json.loads(lines[-1])
    report["bundle_sha256"] = checksum(payload)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    config = json.loads((ROOT / "deploy/release.json").read_text(encoding="utf-8"))
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist" / config["asset"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle, args.output), indent=2))
