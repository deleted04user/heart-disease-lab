import pandas as pd
import streamlit as st

from heart_disease import config as settings

from .common import compatible_report, figure, header


def evaluation_page(report, metadata):
    header(
        "PERFORMANCE & RELIABILITY",
        "Evaluate the evidence",
        "Compare training validation, the historical Cleveland audit, and a separately sourced cohort.",
    )
    if not compatible_report(report, metadata):
        return
    selected = report["selected_model"]
    metrics, config = report["test_results"][selected], report["configuration"]
    st.success(f"Selected from training CV: {selected}")
    st.caption(
        f"{config['train_rows']} training records · {config.get('cv_repeats', 1)} × {config['cv_folds']}-fold stratified CV · {config['test_rows']} Cleveland holdout records · seed {config['seed']} · threshold 0.5"
    )
    if report.get("test_interpretation"):
        st.warning(report["test_interpretation"])
    for columns, keys in [
        (st.columns(3), ["accuracy", "precision", "recall"]),
        (st.columns(3), ["f1", "specificity", "roc_auc"]),
    ]:
        for column, key in zip(columns, keys, strict=True):
            column.metric(
                {"recall": "Recall / sensitivity", "f1": "F1-score", "roc_auc": "ROC-AUC"}.get(
                    key, key.title()
                ),
                f"{metrics[key]:.3f}" if metrics[key] is not None else "Unavailable",
            )
    st.subheader("Training cross-validation comparison")
    comparison_path = settings.REPORT_DIR / "model_comparison.csv"
    if comparison_path.exists():
        try:
            comparison = pd.read_csv(comparison_path)
            if set(comparison["run_id"]) != {report["run_id"]}:
                raise ValueError("comparison belongs to another run")
            labels = {
                "model": "Model",
                "selected": "Selected",
                "cv_roc_auc_mean": "Base CV ROC-AUC",
                "cv_roc_auc_std": "Fold SD",
                "cv_recall_mean": "CV recall",
                "cv_precision_mean": "CV precision",
                "test_roc_auc": "Audit ROC-AUC",
                "test_accuracy": "Audit accuracy",
                "probability_method": "Probability method",
            }
            display = comparison[[col for col in labels if col in comparison]].rename(
                columns=labels
            )
            formats = {
                label: st.column_config.NumberColumn(format="%.3f")
                for col, label in labels.items()
                if col.startswith(("cv_", "test_"))
            }
            st.dataframe(display, column_config=formats, hide_index=True, width="stretch")
            st.download_button(
                "Download complete comparison CSV",
                comparison.to_csv(index=False),
                "model_comparison.csv",
                "text/csv",
            )
            with st.expander("All CV means, standard deviations and test metrics"):
                st.dataframe(comparison, hide_index=True)
        except (ValueError, KeyError, OSError) as exc:
            st.warning(f"Comparison unavailable: {exc}. Regenerate reports.")
    else:
        st.info("Comparison CSV missing; final metrics above come from evaluation_report.json.")
    st.caption(
        config["selection"]
        + " CV ranks raw base classifiers; the selected model's audit metrics use its retained probability method. Repeated folds are correlated; fold SD is not a confidence interval."
    )
    for name, reason in report.get("unavailable", {}).items():
        st.warning(f"{name} unavailable: {reason}")
    a, b = st.columns(2, gap="large")
    with a:
        figure("confusion_matrix.png")
    with b:
        figure("roc_curves.png")
    with st.container(border=True):
        st.markdown(
            f"**{metrics['fp']} false positives:** absence-labelled test records predicted positive. **{metrics['fn']} false negatives:** presence-labelled test records predicted negative."
        )
        st.write(
            "Accuracy alone hides the balance of these errors and changes with class prevalence. ROC-AUC describes ranking, while recall and precision describe this fixed threshold. None establishes clinical safety."
        )
    with st.expander("Class-wise results and experiment configuration"):
        st.dataframe(
            pd.DataFrame(
                {k: v for k, v in metrics["class_report"].items() if isinstance(v, dict)}
            ).T
        )
        st.json(config)
    st.warning(
        f"Only {config['test_rows']} Cleveland holdout records. Estimates can vary substantially with the split; inspect uncertainty and the external check below."
    )
    research_panels(report)


def research_panels(report):
    if "uncertainty" not in report:
        st.info(
            "Run the updated training command to generate uncertainty, probability and subgroup reports."
        )
        return
    st.subheader("Beyond point estimates")
    interval_tab, probability_tab, subgroup_tab, external_tab = st.tabs(
        ["Uncertainty", "Probability quality", "Demographic slices", "External cohort"]
    )
    with interval_tab:
        figure("metric_intervals.png")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Metric": key, **value}
                    for key, value in report["uncertainty"]["metrics"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(report["uncertainty"]["caution"])
        repeats_path = settings.REPORT_DIR / "cv_repeat_summary.csv"
        if repeats_path.exists():
            st.markdown("**Repeated training validation**")
            repeats = pd.read_csv(repeats_path)
            st.line_chart(
                repeats.pivot(index="repeat", columns="model", values="roc_auc"), height=230
            )
            st.caption(
                "Each point averages one repeat's folds. Repeats overlap and are not independent samples for a significance test."
            )
    with probability_tab:
        calibration = report["calibration"]
        st.markdown(f"**Retained probability method: {calibration['selected_method']}**")
        st.dataframe(pd.DataFrame(calibration["summary"]), hide_index=True, width="stretch")
        st.caption(calibration["selection_rule"])
        figure("training_reliability.png")
        st.caption(calibration["caution"])
        cols = st.columns(2)
        cols[0].metric(
            "Cleveland audit Brier score", f"{report['probability_scores']['brier']:.4f}"
        )
        cols[1].metric(
            "Cleveland audit log loss", f"{report['probability_scores']['log_loss']:.4f}"
        )
        st.caption(
            "Lower scores indicate better aggregate probability quality; neither score isolates calibration. No clinical probability validation is implied."
        )
        with st.expander("Cleveland audit reliability diagram"):
            figure("holdout_reliability.png")
    with subgroup_tab:
        st.write(
            "Predeclared recorded-sex and age groups on the Cleveland audit holdout. No fitting or threshold selection uses these slices."
        )
        slices = pd.DataFrame(report["subgroups"])
        st.dataframe(
            slices[
                [
                    "attribute",
                    "group",
                    "n",
                    "positive_n",
                    "negative_n",
                    "accuracy",
                    "recall",
                    "specificity",
                    "roc_auc",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        with st.expander("Wilson intervals, undefined metrics and sample-size cautions"):
            st.dataframe(slices, hide_index=True, width="stretch")
        st.download_button(
            "Download subgroup results",
            slices.to_csv(index=False),
            "subgroup_evaluation.csv",
            "text/csv",
        )
        st.warning(
            "Small and single-class subgroups can have wide or undefined estimates. These slices do not establish fairness, equivalence, or clinical safety. Missing denominators are left undefined."
        )
    with external_tab:
        external = report.get("external", {})
        if external.get("status") != "evaluated":
            st.info(external.get("reason", "No external evaluation is available."))
            st.code(
                "python scripts/download_external.py\npython scripts/train.py --external data/raw/processed.hungarian.data",
                language="shell",
            )
            return
        st.markdown(
            f"**{external['cohort']}** · {external['data']['raw_rows']} source rows → {external['rows_evaluated']} evaluated"
        )
        em = external["metrics"]
        cols = st.columns(3)
        for col, key, label in zip(
            cols,
            ["accuracy", "recall", "roc_auc"],
            ["External accuracy", "External recall", "External ROC-AUC"],
            strict=True,
        ):
            col.metric(label, f"{em[key]:.3f}" if em[key] is not None else "Undefined")
        st.caption(
            f"Confusion matrix [[TN, FP], [FN, TP]]: {em['confusion_matrix']}. Frozen Cleveland-trained preprocessing; no fitting on this cohort."
        )
        st.warning(external["caution"])
        missing = pd.DataFrame(
            [
                {"Feature": name, "Missing": entry["count"], "Missing fraction": entry["fraction"]}
                for name, entry in external["missingness"].items()
            ]
        ).sort_values("Missing", ascending=False)
        st.dataframe(missing, hide_index=True, width="stretch")
        figure("external_reliability.png")
        with st.expander("External intervals, demographic slices and complete source audit"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": key, **value}
                        for key, value in external["uncertainty"]["metrics"].items()
                    ]
                ),
                hide_index=True,
            )
            st.dataframe(pd.DataFrame(external["subgroups"]), hide_index=True)
            st.json(
                {
                    key: external[key]
                    for key in [
                        "data",
                        "exclusions",
                        "overlapping_normalized_row_indices_excluded",
                        "outside_cleveland_observed_ranges",
                        "fitting_on_external_data",
                    ]
                }
            )
