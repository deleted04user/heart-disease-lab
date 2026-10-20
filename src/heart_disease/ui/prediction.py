import json
import os
from pathlib import Path

import streamlit as st

from heart_disease import config as settings
from heart_disease.demo import public_demo_record, result_export
from heart_disease.prediction import ArtifactError, load_artifact, metadata_path, predict_records

from .common import header, recover


@st.cache_resource(show_spinner=False)
def cached_model(path: str, model_stamp: int, metadata_stamp: int):
    # Cache only the trusted fitted model; no patient input/result caching.
    return load_artifact(Path(path))


def prediction_page():
    public_mode = os.environ.get("HEART_PUBLIC_DEMO") == "1"
    header(
        "PUBLIC STUDY DEMONSTRATION" if public_mode else "LOCAL MODEL INFERENCE",
        "Explore a model prediction",
        "Load a bundled public UCI example. Personal measurement entry is disabled."
        if public_mode
        else "Enter all 13 measurements using the original UCI feature meanings.",
    )
    st.warning(settings.DISCLAIMER)
    if public_mode:
        st.info(
            "Public educational demo: use the supplied anonymous study examples only. Do not enter personal or clinical information."
        )
    if not settings.MODEL_PATH.exists() or not metadata_path(settings.MODEL_PATH).exists():
        recover("The saved model is missing. Train on real data before predicting.")
        return
    try:
        pipeline, metadata = cached_model(
            str(settings.MODEL_PATH),
            settings.MODEL_PATH.stat().st_mtime_ns,
            metadata_path(settings.MODEL_PATH).stat().st_mtime_ns,
        )
    except (ArtifactError, OSError) as exc:
        st.error(str(exc))
        return
    st.caption(
        f"Saved model: {metadata['selected_model']} · Values start blank. Entry bounds are broad software validation limits, not medical reference intervals. Use study examples without identifying information."
    )

    def reset_form():
        for field in settings.FEATURES:
            st.session_state[field] = None
        for key in ["prediction_result", "prediction_export", "public_example", "demo_error"]:
            st.session_state.pop(key, None)

    def load_example():
        try:
            record, label = public_demo_record(
                metadata, int(st.session_state.get("demo_example_index", 0))
            )
            reset_form()
            for field, value in record.items():
                st.session_state[field] = value
            st.session_state["public_example"] = label
        except (ValueError, OSError) as exc:
            st.session_state["demo_error"] = str(exc)

    example_count = metadata.get("public_demo", {}).get("count", 1)
    if example_count > 1:
        st.selectbox(
            "Bundled public study example",
            list(range(example_count)),
            format_func=lambda value: f"Public example {value + 1}",
            key="demo_example_index",
            on_change=reset_form,
        )
    demo_columns = st.columns(2)
    demo_columns[0].button(
        "Load public dataset example", key="load_demo", on_click=load_example, width="stretch"
    )
    demo_columns[1].button(
        "Reset all inputs", key="reset_inputs", on_click=reset_form, width="stretch"
    )
    if "public_example" in st.session_state:
        st.info(
            st.session_state["public_example"]
            + (
                " Manual measurement entry is disabled."
                if public_mode
                else " You may edit these values; no input history is saved."
            )
        )
    if "demo_error" in st.session_state:
        st.error(st.session_state["demo_error"])
    values = {}

    def number(col):
        low, high = settings.BOUNDS[col]
        values[col] = st.number_input(
            settings.FIELD_LABELS[col],
            min_value=low,
            max_value=high,
            value=None,
            step=0.1 if col == "oldpeak" else 1,
            key=col,
            placeholder="Load a public example" if public_mode else "Enter a measurement",
            disabled=public_mode,
            help=f"Accepted input bounds: {low} to {high}. Training range: {metadata['data']['observed_ranges'][col]['min']} to {metadata['data']['observed_ranges'][col]['max']}.",
        )

    def category(col):
        values[col] = st.selectbox(
            settings.FIELD_LABELS[col],
            list(settings.CATEGORY_LABELS[col]),
            format_func=lambda code: settings.CATEGORY_LABELS[col][code],
            index=None,
            placeholder="Select a recorded category",
            key=col,
            disabled=public_mode,
        )

    with st.form("prediction_form", clear_on_submit=False):
        a, b = st.columns(2, gap="large")
        with a:
            st.subheader("Basic information & blood measurements")
            number("age")
            category("sex")
            number("trestbps")
            number("chol")
            category("fbs")
            st.subheader("Additional indicators")
            category("ca")
            category("thal")
        with b:
            st.subheader("Cardiac & exercise measurements")
            category("cp")
            category("restecg")
            number("thalach")
            category("exang")
            number("oldpeak")
            category("slope")
        submitted = st.form_submit_button(
            "Run model prediction", key="predict_submit", type="primary", width="stretch"
        )
    st.caption(
        "The hosted Python app processes bundled public study records only. No input history or database is stored; no external prediction API is used."
        if public_mode
        else "Inputs are processed by this local Python application. They are not saved to files, logged, or sent to external prediction services. The browser session temporarily holds widget values."
    )
    if submitted:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_export", None)
        missing = [settings.FIELD_LABELS[col] for col in settings.FEATURES if values[col] is None]
        if missing:
            st.error("Complete all fields: " + ", ".join(missing))
            return
        try:
            if public_mode:
                approved, _ = public_demo_record(
                    metadata, int(st.session_state.get("demo_example_index", 0))
                )
                if any(
                    value is None or abs(float(value) - float(approved[field])) > 1e-6
                    for field, value in values.items()
                ):
                    raise ValueError(
                        "The public demo accepts only the currently selected bundled study record."
                    )
            result = predict_records(pipeline, values).iloc[0]
        except (ValueError, RuntimeError) as exc:
            st.error(f"Prediction failed: {exc}")
            return
        st.session_state["prediction_result"] = result.to_dict()
        st.session_state["prediction_export"] = result_export(
            result.to_dict(), metadata, example_loaded="public_example" in st.session_state
        )
        outside = [
            col
            for col in settings.BOUNDS
            if not metadata["data"]["observed_ranges"][col]["min"]
            <= values[col]
            <= metadata["data"]["observed_ranges"][col]["max"]
        ]
        if outside:
            st.warning(
                "Outside observed training ranges: "
                + ", ".join(outside)
                + ". This output may be unreliable."
            )
    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]
        positive = int(result["prediction"]) == 1
        with st.container(border=True):
            st.subheader(
                "Model output · " + ("positive class (1)" if positive else "negative class (0)")
            )
            st.write(
                "The model assigned the input to the dataset's **"
                + ("disease-presence" if positive else "disease-absence")
                + "** label. This does not confirm or exclude a medical condition."
            )
            if "positive_class_probability" in result:
                probability = float(result["positive_class_probability"])
                st.metric(
                    "Model-estimated probability for the positive class", f"{probability:.1%}"
                )
                st.progress(probability)
                st.caption(
                    f"Probability processing: {metadata.get('probability_method', 'raw')} · fixed threshold 0.5. This is not a clinically validated risk score or a prediction of future disease."
                )
            st.download_button(
                "Download prediction summary (JSON)",
                json.dumps(st.session_state["prediction_export"], indent=2),
                "prediction_summary.json",
                "application/json",
                key="download_prediction",
            )
            st.caption(
                "The download contains model output and run information, without the entered measurements or identifying information."
            )
