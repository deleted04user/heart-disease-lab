"""Exercise the pinned cloud entrypoint in a fresh serving-only Python process."""

import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "deploy/streamlit_app.py")).run(timeout=60)
    assert not app.exception and not app.error, "Cloud bootstrap did not load"
    for page in [
        "Model Evaluation",
        "Explainability",
        "About & Limitations",
        "Patient Input & Prediction",
    ]:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception and not app.error, page
    module = sys.modules["heart_disease.config"]
    assert module.MODEL_PATH.resolve().is_relative_to(root / ".deployment-cache")
    assert all(widget.disabled for widget in app.number_input)
    assert all(widget.disabled for widget in app.selectbox if widget.key != "demo_example_index")
    from heart_disease.demo import public_demo_record
    from heart_disease.prediction import load_artifact, predict_records

    pipeline, metadata = load_artifact(module.MODEL_PATH)
    for index in (0, metadata["public_demo"]["count"] - 1):
        app.selectbox(key="demo_example_index").set_value(index).run(timeout=30)
        app.button(key="load_demo").click().run(timeout=30)
        app.button(key="predict_submit").click().run(timeout=30)
        assert not app.exception and not app.error
        record, _ = public_demo_record(metadata, index)
        expected = predict_records(pipeline, record).iloc[0].positive_class_probability
        assert any(metric.value == f"{expected:.1%}" for metric in app.metric)
        assert app.session_state["prediction_export"]["public_example_loaded"] is True
    print(
        json.dumps(
            {
                "status": "passed",
                "run_id": metadata["run_id"],
                "checks": [
                    "pinned cloud bootstrap",
                    "all five pages",
                    "disabled personal inputs",
                    "two public predictions match the saved model",
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
