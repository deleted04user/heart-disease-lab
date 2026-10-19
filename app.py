"""Heart Disease Lab: local, privacy-conscious educational Streamlit dashboard."""
import sys
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from heart_disease.config import (
    BOUNDS, CATEGORY_LABELS, DATA_PATH, DISCLAIMER, FEATURES, FIELD_LABELS,
    MODEL_PATH, REPORT_DIR, UCI_PAGE,
)
from heart_disease.data import load_dataset
from heart_disease.io import read_json
from heart_disease.prediction import ArtifactError, load_artifact, metadata_path, predict_records
from heart_disease.demo import public_demo_record, result_export
from heart_disease.logging_config import configure_logging

st.set_page_config(page_title="Heart Disease Lab", page_icon="♥", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
.block-container {max-width: 1260px; padding-top: 3.8rem; padding-bottom: 3rem;}
h1 {letter-spacing: -0.045em; font-weight: 750 !important;}
h2, h3 {letter-spacing: -0.02em;}
[data-testid="stSidebar"] {border-right: 1px solid #E1E7F0;}
[data-testid="stMetric"] {background: white; border: 1px solid #E1E7F0;
    border-radius: 12px; padding: 18px 20px;}
[data-testid="stMetricLabel"] {color: #536987;}
[data-testid="stMetricValue"] {font-weight: 650; color: #172B4D;}
.eyebrow {color: #2463EB; font-size: 0.77rem; font-weight: 700;
    letter-spacing: 0.14em; margin-bottom: 0.5rem;}
.brand {font-size: 1.42rem; font-weight: 750; letter-spacing: -0.04em; margin: 0.5rem 0;}
.subtle {color: #64748B; font-size: 0.86rem;}
div[data-testid="stForm"] {background: white; border-color: #E1E7F0; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


def json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (ValueError, OSError) as exc:
        st.warning(f"Cannot read {path.name}: {exc}. Recreate reports with python scripts/train.py.")
        return None


def figure(name: str, caption: str = ""):
    path = REPORT_DIR / "figures" / name
    if path.exists():
        try:
            st.image(str(path), caption=caption or None, width="stretch")
        except Exception:
            st.info(f"Figure {name} cannot be displayed. Regenerate it with python scripts/train.py.")
    else:
        st.info(f"Figure {name} is unavailable. Run python scripts/train.py.")


def header(kicker: str, title: str, subtitle: str):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(subtitle)


def recover(message: str):
    st.info(message)
    st.code("python scripts/download_data.py\npython scripts/train.py", language="powershell")


def compatible_report(report: dict | None, metadata: dict | None) -> bool:
    if not report or not {"run_id", "selected_model", "test_results", "configuration", "data"}.issubset(report):
        recover("No complete evaluation report is available yet.")
        return False
    if metadata and report.get("run_id") != metadata.get("run_id"):
        st.error("Model and report belong to different runs. Rerun training with matching model/report paths.")
        return False
    return True


def overview(report, metadata):
    header("REPRODUCIBLE MACHINE LEARNING", "Heart disease, explored through data.", "A transparent journey from historical observations to a tested classification model.")
    st.info(DISCLAIMER)
    data = (report or metadata or {}).get("data")
    eda = json_file(REPORT_DIR / "eda_summary.json")
    if not data:
        data = (eda or {}).get("data")
    if not data and DATA_PATH.exists():
        try:
            data = load_dataset(DATA_PATH).metadata
        except (ValueError, OSError) as exc:
            st.error(f"Dataset validation failed: {exc}")
    if not data:
        recover("Get the real UCI data and train locally to populate this dashboard.")
        return
    counts = data["target_distribution"]
    columns = st.columns(4)
    for column, label, value in zip(columns, ["Validated records", "Input features", "Positive class", "Missing cells"], [f"{data['rows']:,}", data["feature_count"], f"{int(counts['1']) / data['rows']:.1%}", sum(data["missing_values"].values())]):
        column.metric(label, value)
    st.caption(f"Training-source snapshot · {data['variant']}")
    st.markdown(f"[UCI Heart Disease dataset]({UCI_PAGE}) · {data['source']['source']}. Target mapping: {data['target_mapping']}.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("What the records show")
        st.bar_chart(pd.DataFrame({"Records": [int(counts['0']), int(counts['1'])]}, index=["Absence · 0", "Presence · 1"]), color="#2463EB", height=260)
        st.caption("Labels describe the recorded disease status in this dataset; they do not measure future disease incidence.")
    with right:
        st.subheader("Data quality at a glance")
        missing = pd.DataFrame({"Feature": list(data["missing_values"]), "Missing": list(data["missing_values"].values())})
        st.dataframe(missing.loc[missing["Missing"] > 0], hide_index=True, width="stretch")
        st.write(f"**{data['duplicates_removed']}** exact duplicates removed before splitting. **{data['records_corrected']}** values automatically corrected.")
        st.caption("Missing feature values are imputed inside training folds. Missing targets and invalid source codes are rejected.")
        for note in data.get("notes", []):
            st.caption(note)
    st.subheader("From source to reproducible output")
    stages = st.columns(4)
    for col, title, text in zip(stages, ["01 · Validate", "02 · Compare", "03 · Evaluate", "04 · Explain"], ["One source, explicit schema and a recorded checksum.", "Fold-local pipelines and training-only model selection.", "A stratified holdout, six metrics and class-wise errors.", "Held-out permutation importance for the 13 original inputs."]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(text)
    if eda and eda.get("data", {}).get("sha256") == data.get("sha256"):
        with st.expander("Explore the measurements and relationships"):
            a, b = st.columns(2)
            with a:
                figure("numeric_distributions.png")
            with b:
                figure("correlations.png", "Correlations describe associations, not causation.")
            figure("categorical_distributions.png")
    with st.expander("Source details and audit trail"):
        st.json(data)


@st.cache_resource(show_spinner=False)
def cached_model(path: str, model_stamp: int, metadata_stamp: int):
    # Cache only the trusted fitted model; no patient input/result caching.
    return load_artifact(Path(path))


def prediction_page():
    public_mode = os.environ.get("HEART_PUBLIC_DEMO") == "1"
    header("PUBLIC STUDY DEMONSTRATION" if public_mode else "LOCAL MODEL INFERENCE", "Explore a model prediction", "Load a bundled public UCI example. Personal measurement entry is disabled." if public_mode else "Enter all 13 measurements using the original UCI feature meanings.")
    st.warning(DISCLAIMER)
    if public_mode:
        st.info("Public educational demo: use the supplied anonymous study examples only. Do not enter personal or clinical information.")
    if not MODEL_PATH.exists() or not metadata_path(MODEL_PATH).exists():
        recover("The saved model is missing. Train on real data before predicting.")
        return
    try:
        pipeline, metadata = cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns, metadata_path(MODEL_PATH).stat().st_mtime_ns)
    except (ArtifactError, OSError) as exc:
        st.error(str(exc))
        return
    st.caption(f"Saved model: {metadata['selected_model']} · Values start blank. Entry bounds are broad software validation limits, not medical reference intervals. Use study examples without identifying information.")
    def reset_form():
        for field in FEATURES:
            st.session_state[field] = None
        for key in ["prediction_result", "prediction_export", "public_example", "demo_error"]:
            st.session_state.pop(key, None)

    def load_example():
        try:
            record, label = public_demo_record(metadata, int(st.session_state.get("demo_example_index", 0)))
            reset_form()
            for field, value in record.items():
                st.session_state[field] = value
            st.session_state["public_example"] = label
        except (ValueError, OSError) as exc:
            st.session_state["demo_error"] = str(exc)

    example_count = metadata.get("public_demo", {}).get("count", 1)
    if example_count > 1:
        st.selectbox("Bundled public study example", list(range(example_count)), format_func=lambda value: f"Public example {value + 1}", key="demo_example_index", on_change=reset_form)
    demo_columns = st.columns(2)
    demo_columns[0].button("Load public dataset example", key="load_demo", on_click=load_example, width="stretch")
    demo_columns[1].button("Reset all inputs", key="reset_inputs", on_click=reset_form, width="stretch")
    if "public_example" in st.session_state:
        st.info(st.session_state["public_example"] + (" Manual measurement entry is disabled." if public_mode else " You may edit these values; no input history is saved."))
    if "demo_error" in st.session_state:
        st.error(st.session_state["demo_error"])
    values = {}

    def number(col):
        low, high = BOUNDS[col]
        values[col] = st.number_input(FIELD_LABELS[col], min_value=low, max_value=high, value=None, step=0.1 if col == "oldpeak" else 1, key=col, placeholder="Load a public example" if public_mode else "Enter a measurement", disabled=public_mode, help=f"Accepted input bounds: {low} to {high}. Training range: {metadata['data']['observed_ranges'][col]['min']} to {metadata['data']['observed_ranges'][col]['max']}.")

    def category(col):
        values[col] = st.selectbox(FIELD_LABELS[col], list(CATEGORY_LABELS[col]), format_func=lambda code: CATEGORY_LABELS[col][code], index=None, placeholder="Select a recorded category", key=col, disabled=public_mode)

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
        submitted = st.form_submit_button("Run model prediction", key="predict_submit", type="primary", width="stretch")
    st.caption("The hosted Python app processes bundled public study records only. No input history or database is stored; no external prediction API is used." if public_mode else "Inputs are processed by this local Python application. They are not saved to files, logged, or sent to external prediction services. The browser session temporarily holds widget values.")
    if submitted:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_export", None)
        missing = [FIELD_LABELS[col] for col in FEATURES if values[col] is None]
        if missing:
            st.error("Complete all fields: " + ", ".join(missing))
            return
        try:
            if public_mode:
                approved, _ = public_demo_record(metadata, int(st.session_state.get("demo_example_index", 0)))
                if any(abs(float(values[field]) - float(approved[field])) > 1e-6 for field in FEATURES):
                    raise ValueError("The public demo accepts only the currently selected bundled study record.")
            result = predict_records(pipeline, values).iloc[0]
        except (ValueError, RuntimeError) as exc:
            st.error(f"Prediction failed: {exc}")
            return
        st.session_state["prediction_result"] = result.to_dict()
        st.session_state["prediction_export"] = result_export(result.to_dict(), metadata, example_loaded="public_example" in st.session_state)
        outside = [col for col in BOUNDS if not metadata['data']['observed_ranges'][col]['min'] <= values[col] <= metadata['data']['observed_ranges'][col]['max']]
        if outside:
            st.warning("Outside observed training ranges: " + ", ".join(outside) + ". This output may be unreliable.")
    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]
        positive = int(result["prediction"]) == 1
        with st.container(border=True):
            st.subheader("Model output · " + ("positive class (1)" if positive else "negative class (0)"))
            st.write("The model assigned the input to the dataset's **" + ("disease-presence" if positive else "disease-absence") + "** label. This does not confirm or exclude a medical condition.")
            if "positive_class_probability" in result:
                probability = float(result["positive_class_probability"])
                st.metric("Model-estimated probability for the positive class", f"{probability:.1%}")
                st.progress(probability)
                st.caption(f"Probability processing: {metadata.get('probability_method', 'raw')} · fixed threshold 0.5. This is not a clinically validated risk score or a prediction of future disease.")
            st.download_button("Download prediction summary (JSON)", json.dumps(st.session_state["prediction_export"], indent=2), "prediction_summary.json", "application/json", key="download_prediction")
            st.caption("The download contains model output and run information, without the entered measurements or identifying information.")


def evaluation_page(report, metadata):
    header("PERFORMANCE & RELIABILITY", "Evaluate the evidence", "Compare training validation, the historical Cleveland audit, and a separately sourced cohort.")
    if not compatible_report(report, metadata):
        return
    selected = report["selected_model"]
    metrics, config = report["test_results"][selected], report["configuration"]
    st.success(f"Selected from training CV: {selected}")
    st.caption(f"{config['train_rows']} training records · {config.get('cv_repeats', 1)} × {config['cv_folds']}-fold stratified CV · {config['test_rows']} Cleveland holdout records · seed {config['seed']} · threshold 0.5")
    if report.get("test_interpretation"):
        st.warning(report["test_interpretation"])
    for columns, keys in [(st.columns(3), ["accuracy", "precision", "recall"]), (st.columns(3), ["f1", "specificity", "roc_auc"])]:
        for column, key in zip(columns, keys):
            column.metric({"recall": "Recall / sensitivity", "f1": "F1-score", "roc_auc": "ROC-AUC"}.get(key, key.title()), f"{metrics[key]:.3f}" if metrics[key] is not None else "Unavailable")
    st.subheader("Training cross-validation comparison")
    comparison_path = REPORT_DIR / "model_comparison.csv"
    if comparison_path.exists():
        try:
            comparison = pd.read_csv(comparison_path)
            if set(comparison["run_id"]) != {report["run_id"]}:
                raise ValueError("comparison belongs to another run")
            labels = {"model": "Model", "selected": "Selected", "cv_roc_auc_mean": "Base CV ROC-AUC", "cv_roc_auc_std": "Fold SD", "cv_recall_mean": "CV recall", "cv_precision_mean": "CV precision", "test_roc_auc": "Audit ROC-AUC", "test_accuracy": "Audit accuracy", "probability_method": "Probability method"}
            display = comparison[[col for col in labels if col in comparison]].rename(columns=labels)
            formats = {label: st.column_config.NumberColumn(format="%.3f") for col, label in labels.items() if col.startswith(("cv_", "test_"))}
            st.dataframe(display, column_config=formats, hide_index=True, width="stretch")
            st.download_button("Download complete comparison CSV", comparison.to_csv(index=False), "model_comparison.csv", "text/csv")
            with st.expander("All CV means, standard deviations and test metrics"):
                st.dataframe(comparison, hide_index=True)
        except (ValueError, KeyError, OSError) as exc:
            st.warning(f"Comparison unavailable: {exc}. Regenerate reports.")
    else:
        st.info("Comparison CSV missing; final metrics above come from evaluation_report.json.")
    st.caption(config["selection"] + " CV ranks raw base classifiers; the selected model's audit metrics use its retained probability method. Repeated folds are correlated; fold SD is not a confidence interval.")
    for name, reason in report.get("unavailable", {}).items():
        st.warning(f"{name} unavailable: {reason}")
    a, b = st.columns(2, gap="large")
    with a:
        figure("confusion_matrix.png")
    with b:
        figure("roc_curves.png")
    with st.container(border=True):
        st.markdown(f"**{metrics['fp']} false positives:** absence-labelled test records predicted positive. **{metrics['fn']} false negatives:** presence-labelled test records predicted negative.")
        st.write("Accuracy alone hides the balance of these errors and changes with class prevalence. ROC-AUC describes ranking, while recall and precision describe this fixed threshold. None establishes clinical safety.")
    with st.expander("Class-wise results and experiment configuration"):
        st.dataframe(pd.DataFrame({k: v for k, v in metrics["class_report"].items() if isinstance(v, dict)}).T)
        st.json(config)
    st.warning(f"Only {config['test_rows']} Cleveland holdout records. Estimates can vary substantially with the split; inspect uncertainty and the external check below.")
    research_panels(report)


def research_panels(report):
    if "uncertainty" not in report:
        st.info("Run the updated training command to generate uncertainty, probability and subgroup reports.")
        return
    st.subheader("Beyond point estimates")
    interval_tab, probability_tab, subgroup_tab, external_tab = st.tabs(["Uncertainty", "Probability quality", "Demographic slices", "External cohort"])
    with interval_tab:
        figure("metric_intervals.png")
        st.dataframe(pd.DataFrame([{"Metric": key, **value} for key, value in report["uncertainty"]["metrics"].items()]), hide_index=True, width="stretch")
        st.caption(report["uncertainty"]["caution"])
        repeats_path = REPORT_DIR / "cv_repeat_summary.csv"
        if repeats_path.exists():
            st.markdown("**Repeated training validation**")
            repeats = pd.read_csv(repeats_path)
            st.line_chart(repeats.pivot(index="repeat", columns="model", values="roc_auc"), height=230)
            st.caption("Each point averages one repeat's folds. Repeats overlap and are not independent samples for a significance test.")
    with probability_tab:
        calibration = report["calibration"]
        st.markdown(f"**Retained probability method: {calibration['selected_method']}**")
        st.dataframe(pd.DataFrame(calibration["summary"]), hide_index=True, width="stretch")
        st.caption(calibration["selection_rule"])
        figure("training_reliability.png")
        st.caption(calibration["caution"])
        cols = st.columns(2)
        cols[0].metric("Cleveland audit Brier score", f"{report['probability_scores']['brier']:.4f}")
        cols[1].metric("Cleveland audit log loss", f"{report['probability_scores']['log_loss']:.4f}")
        st.caption("Lower scores indicate better aggregate probability quality; neither score isolates calibration. No clinical probability validation is implied.")
        with st.expander("Cleveland audit reliability diagram"):
            figure("holdout_reliability.png")
    with subgroup_tab:
        st.write("Predeclared recorded-sex and age groups on the Cleveland audit holdout. No fitting or threshold selection uses these slices.")
        slices = pd.DataFrame(report["subgroups"])
        st.dataframe(slices[["attribute", "group", "n", "positive_n", "negative_n", "accuracy", "recall", "specificity", "roc_auc"]], hide_index=True, width="stretch")
        with st.expander("Wilson intervals, undefined metrics and sample-size cautions"):
            st.dataframe(slices, hide_index=True, width="stretch")
        st.download_button("Download subgroup results", slices.to_csv(index=False), "subgroup_evaluation.csv", "text/csv")
        st.warning("Small and single-class subgroups can have wide or undefined estimates. These slices do not establish fairness, equivalence, or clinical safety. Missing denominators are left undefined.")
    with external_tab:
        external = report.get("external", {})
        if external.get("status") != "evaluated":
            st.info(external.get("reason", "No external evaluation is available."))
            st.code("python scripts/download_external.py\npython scripts/train.py --external data/raw/processed.hungarian.data", language="shell")
            return
        st.markdown(f"**{external['cohort']}** · {external['data']['raw_rows']} source rows → {external['rows_evaluated']} evaluated")
        em = external["metrics"]
        cols = st.columns(3)
        for col, key, label in zip(cols, ["accuracy", "recall", "roc_auc"], ["External accuracy", "External recall", "External ROC-AUC"]):
            col.metric(label, f"{em[key]:.3f}" if em[key] is not None else "Undefined")
        st.caption(f"Confusion matrix [[TN, FP], [FN, TP]]: {em['confusion_matrix']}. Frozen Cleveland-trained preprocessing; no fitting on this cohort.")
        st.warning(external["caution"])
        missing = pd.DataFrame([{"Feature": name, "Missing": entry["count"], "Missing fraction": entry["fraction"]} for name, entry in external["missingness"].items()]).sort_values("Missing", ascending=False)
        st.dataframe(missing, hide_index=True, width="stretch")
        figure("external_reliability.png")
        with st.expander("External intervals, demographic slices and complete source audit"):
            st.dataframe(pd.DataFrame([{"Metric": key, **value} for key, value in external["uncertainty"]["metrics"].items()]), hide_index=True)
            st.dataframe(pd.DataFrame(external["subgroups"]), hide_index=True)
            st.json({key: external[key] for key in ["data", "exclusions", "overlapping_normalized_row_indices_excluded", "outside_cleveland_observed_ranges", "fitting_on_external_data"]})


def explainability_page(report, metadata):
    header("MODEL INTERPRETATION", "Understand the inputs", "Measure how shuffling an original feature changes this fitted model's held-out ROC-AUC.")
    if not compatible_report(report, metadata):
        return
    explanation = json_file(REPORT_DIR / "explainability.json")
    if not explanation or explanation.get("run_id") != report["run_id"]:
        recover("Matching explainability results are unavailable. Training recreates them from the Cleveland audit holdout.")
        return
    figure("permutation_importance.png")
    table = pd.DataFrame(explanation["features"])
    st.dataframe(table, hide_index=True, width="stretch")
    leading = ", ".join(table.head(3)["feature"])
    st.write(f"For this run, the largest mean ROC-AUC decreases occur for **{leading}**. Larger decreases indicate that this model relies more on those inputs within these held-out observations.")
    st.caption(f"{explanation['repeats']} shuffles per feature · {explanation['test_rows']} test observations · original features, before one-hot encoding.")
    st.warning(explanation["caution"] + " Importance does not establish causation or clinical importance. Do not use these test-set diagnostics to tune or reselect this experiment.")


def about_page(report, metadata):
    header("METHOD & RESPONSIBLE USE", "A small dataset. Clear boundaries.", "An educational portfolio project designed for reproducibility, inspection and honest interpretation.")
    st.warning(DISCLAIMER)
    st.markdown(f"""
**Dataset origin.** [UCI Heart Disease]({UCI_PAGE}), Janosi, Steinbrunn, Pfisterer and Detrano (1989), [DOI](https://doi.org/10.24432/C52P4X), CC BY 4.0. The default is the original processed Cleveland file. Local mirrors require explicit schema and site selection.

**Target.** Original angiographic disease-status codes 0–4 become absence (0) versus presence (1–4 → 1). Already-binary target headers require an explicit assertion that 1 means presence. This is historical recorded status, not future disease incidence.

**Architecture.** Logistic Regression, Random Forest and XGBoost share scikit-learn pipelines. Five numeric features use median imputation (plus scaling for Logistic Regression). Eight categorical features use modal imputation and one-hot encoding. Unknown categories are safe in the encoder; public inference rejects undocumented codes.

**Evaluation.** A stratified 80/20 split with seed 42. Three repeats of training-only stratified CV, up to five folds, select a base classifier by mean ROC-AUC, then recall and precision. Nested training-only checks retain sigmoid calibration only if the declared Brier/log-loss rule passes. The threshold stays 0.5. The previously viewed Cleveland holdout is labelled as an audit. A separate Hungarian-site check is available when explicitly supplied. No oversampling, test-driven tuning, or fitting on either evaluation cohort.

**Limits and potential bias.** Small, historical hospital samples, site selection, measurement differences and missingness constrain generalization. Limited sex codes do not represent every person. Predeclared demographic slices are descriptive and can be very small; no fairness guarantee, prospective study or clinical probability validation is established. The historical site-transfer check can be affected by severe missingness. Feature importance shows model associations only.

**Privacy.** Prediction runs within the local app. No patient database or external inference service is used. Browser session state temporarily holds widget values; avoid entering identifying information. Streamlit usage telemetry is disabled.

**Artifact trust.** Only load joblib files you created or trust; deserialization can execute code. The stored checksum detects corruption and mismatched metadata, not malicious files.
""")
    st.subheader("Run it locally")
    st.code("python -m pip install -r requirements.txt\npython -m pip install -e . --no-deps\npython scripts/download_data.py\npython scripts/download_external.py\npython scripts/train.py --external data/raw/processed.hungarian.data\npython -m pytest\npython -m streamlit run app.py", language="shell")
    st.caption("Create and activate a Python 3.10+ virtual environment first. README.md contains Windows, macOS/Linux and manual-source instructions.")
    card = REPORT_DIR / "model_card.md"
    if card.exists():
        with st.expander("Read the model card"):
            st.markdown(card.read_text(encoding="utf-8"))
        st.download_button("Download model card", card.read_text(encoding="utf-8"), "model_card.md", "text/markdown")
    with st.expander("Model metadata and dependency versions"):
        if metadata:
            st.json(metadata)
        else:
            st.info("Metadata will appear after a successful real-data training run.")


def main():
    configure_logging()
    metadata = json_file(metadata_path(MODEL_PATH))
    report = json_file(REPORT_DIR / "evaluation_report.json")
    with st.sidebar:
        st.markdown('<div class="brand">♥ &nbsp; Heart Disease Lab</div><div class="subtle">Data science · Educational research</div>', unsafe_allow_html=True)
        st.divider()
        page = st.radio("Workspace", ["Project Overview", "Patient Input & Prediction", "Model Evaluation", "Explainability", "About & Limitations"], label_visibility="collapsed")
        st.divider()
        if metadata and MODEL_PATH.exists():
            st.markdown("**Saved experiment**")
            st.caption(metadata.get("selected_model", "Unknown model"))
            st.caption(f"Seed 42 · Run {metadata.get('run_id', 'unknown')[:8]}")
        else:
            st.caption("Awaiting a real-data training run")
        st.markdown(f"[Dataset documentation]({UCI_PAGE})")
        st.caption("Local inference. No external prediction API.")
    if page == "Project Overview":
        overview(report, metadata)
    elif page == "Patient Input & Prediction":
        prediction_page()
    elif page == "Model Evaluation":
        evaluation_page(report, metadata)
    elif page == "Explainability":
        explainability_page(report, metadata)
    else:
        about_page(report, metadata)


if __name__ == "__main__":
    main()
