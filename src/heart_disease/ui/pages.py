import pandas as pd
import streamlit as st

from heart_disease import config as settings
from heart_disease.data import load_dataset

from .common import compatible_report, figure, header, json_file, recover


def overview(report, metadata):
    header(
        "REPRODUCIBLE MACHINE LEARNING",
        "Heart disease, explored through data.",
        "A transparent journey from historical observations to a tested classification model.",
    )
    st.info(settings.DISCLAIMER)
    data = (report or metadata or {}).get("data")
    eda = json_file(settings.REPORT_DIR / "eda_summary.json")
    if not data:
        data = (eda or {}).get("data")
    if not data and settings.DATA_PATH.exists():
        try:
            data = load_dataset(settings.DATA_PATH).metadata
        except (ValueError, OSError) as exc:
            st.error(f"Dataset validation failed: {exc}")
    if not data:
        recover("Get the real UCI data and train locally to populate this dashboard.")
        return
    counts = data["target_distribution"]
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ["Validated records", "Input features", "Positive class", "Missing cells"],
        [
            f"{data['rows']:,}",
            data["feature_count"],
            f"{int(counts['1']) / data['rows']:.1%}",
            sum(data["missing_values"].values()),
        ],
        strict=True,
    ):
        column.metric(label, value)
    st.caption(f"Training-source snapshot · {data['variant']}")
    st.markdown(
        f"[UCI Heart Disease dataset]({settings.UCI_PAGE}) · {data['source']['source']}. Target mapping: {data['target_mapping']}."
    )
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("What the records show")
        st.bar_chart(
            pd.DataFrame(
                {"Records": [int(counts["0"]), int(counts["1"])]},
                index=["Absence · 0", "Presence · 1"],
            ),
            color="#2463EB",
            height=260,
        )
        st.caption(
            "Labels describe the recorded disease status in this dataset; they do not measure future disease incidence."
        )
    with right:
        st.subheader("Data quality at a glance")
        missing = pd.DataFrame(
            {
                "Feature": list(data["missing_values"]),
                "Missing": list(data["missing_values"].values()),
            }
        )
        st.dataframe(missing.loc[missing["Missing"] > 0], hide_index=True, width="stretch")
        st.write(
            f"**{data['duplicates_removed']}** exact duplicates removed before splitting. **{data['records_corrected']}** values automatically corrected."
        )
        st.caption(
            "Missing feature values are imputed inside training folds. Missing targets and invalid source codes are rejected."
        )
        for note in data.get("notes", []):
            st.caption(note)
    st.subheader("From source to reproducible output")
    stages = st.columns(4)
    for col, title, text in zip(
        stages,
        ["01 · Validate", "02 · Compare", "03 · Evaluate", "04 · Explain"],
        [
            "One source, explicit schema and a recorded checksum.",
            "Fold-local pipelines and training-only model selection.",
            "A stratified holdout, six metrics and class-wise errors.",
            "Held-out permutation importance for the 13 original inputs.",
        ],
        strict=True,
    ):
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


def explainability_page(report, metadata):
    header(
        "MODEL INTERPRETATION",
        "Understand the inputs",
        "Measure how shuffling an original feature changes this fitted model's held-out ROC-AUC.",
    )
    if not compatible_report(report, metadata):
        return
    explanation = json_file(settings.REPORT_DIR / "explainability.json")
    if not explanation or explanation.get("run_id") != report["run_id"]:
        recover(
            "Matching explainability results are unavailable. Training recreates them from the Cleveland audit holdout."
        )
        return
    figure("permutation_importance.png")
    table = pd.DataFrame(explanation["features"])
    st.dataframe(table, hide_index=True, width="stretch")
    leading = ", ".join(table.head(3)["feature"])
    st.write(
        f"For this run, the largest mean ROC-AUC decreases occur for **{leading}**. Larger decreases indicate that this model relies more on those inputs within these held-out observations."
    )
    st.caption(
        f"{explanation['repeats']} shuffles per feature · {explanation['test_rows']} test observations · original features, before one-hot encoding."
    )
    st.warning(
        explanation["caution"]
        + " Importance does not establish causation or clinical importance. Do not use these test-set diagnostics to tune or reselect this experiment."
    )


def about_page(report, metadata):
    header(
        "METHOD & RESPONSIBLE USE",
        "A small dataset. Clear boundaries.",
        "An educational portfolio project designed for reproducibility, inspection and honest interpretation.",
    )
    st.warning(settings.DISCLAIMER)
    st.markdown(f"""
**Dataset origin.** [UCI Heart Disease]({settings.UCI_PAGE}), Janosi, Steinbrunn, Pfisterer and Detrano (1989), [DOI](https://doi.org/10.24432/C52P4X), CC BY 4.0. The default is the original processed Cleveland file. Local mirrors require explicit schema and site selection.

**Target.** Original angiographic disease-status codes 0–4 become absence (0) versus presence (1–4 → 1). Already-binary target headers require an explicit assertion that 1 means presence. This is historical recorded status, not future disease incidence.

**Architecture.** Logistic Regression, Random Forest and XGBoost share scikit-learn pipelines. Five numeric features use median imputation (plus scaling for Logistic Regression). Eight categorical features use modal imputation and one-hot encoding. Unknown categories are safe in the encoder; public inference rejects undocumented codes.

**Evaluation.** A stratified 80/20 split with seed 42. Three repeats of training-only stratified CV, up to five folds, select a base classifier by mean ROC-AUC, then recall and precision. Nested training-only checks retain sigmoid calibration only if the declared Brier/log-loss rule passes. The threshold stays 0.5. The previously viewed Cleveland holdout is labelled as an audit. A separate Hungarian-site check is available when explicitly supplied. No oversampling, test-driven tuning, or fitting on either evaluation cohort.

**Limits and potential bias.** Small, historical hospital samples, site selection, measurement differences and missingness constrain generalization. Limited sex codes do not represent every person. Predeclared demographic slices are descriptive and can be very small; no fairness guarantee, prospective study or clinical probability validation is established. The historical site-transfer check can be affected by severe missingness. Feature importance shows model associations only.

**Privacy.** Prediction runs within the local app. No patient database or external inference service is used. Browser session state temporarily holds widget values; avoid entering identifying information. Streamlit usage telemetry is disabled.

**Artifact trust.** Only load joblib files you created or trust; deserialization can execute code. The stored checksum detects corruption and mismatched metadata, not malicious files.
""")
    st.subheader("Run it locally")
    st.code(
        "python -m pip install -r requirements.txt\npython -m pip install -e . --no-deps\npython scripts/download_data.py\npython scripts/download_external.py\npython scripts/train.py --external data/raw/processed.hungarian.data\npython -m pytest\npython -m streamlit run app.py",
        language="shell",
    )
    st.caption(
        "Create and activate a Python 3.10+ virtual environment first. README.md contains Windows, macOS/Linux and manual-source instructions."
    )
    card = settings.REPORT_DIR / "model_card.md"
    if card.exists():
        with st.expander("Read the model card"):
            st.markdown(card.read_text(encoding="utf-8"))
        st.download_button(
            "Download model card",
            card.read_text(encoding="utf-8"),
            "model_card.md",
            "text/markdown",
        )
    with st.expander("Model metadata and dependency versions"):
        if metadata:
            st.json(metadata)
        else:
            st.info("Metadata will appear after a successful real-data training run.")
