"""Binary metrics with explicit labels and undefined-AUC handling."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)


def specificity(y_true, y_pred) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if tn + fp else 0.0


def evaluate(y_true, y_pred, probabilities) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    auc = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity(y_true, y_pred) if tn + fp else None,
        "roc_auc": auc, "confusion_matrix": matrix.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "class_report": classification_report(y_true, y_pred, labels=[0, 1], target_names=["0: absence", "1: presence"], output_dict=True, zero_division=0),
    }


def save_diagnostic_plots(evaluations: dict, predictions: dict, y_test, selected: str, output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(np.asarray(evaluations[selected]["confusion_matrix"]), display_labels=["Absence (0)", "Presence (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{selected} · final test set")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    curves = {}
    for name, probabilities in predictions.items():
        if evaluations[name]["roc_auc"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    save_roc_plot(curves, evaluations, output_dir / "roc_curves.png")
    from .io import write_json
    write_json(output_dir.parent / "roc_curves.json", curves)


def save_roc_plot(curves: dict, evaluations: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"Logistic Regression": "#2463EB", "Random Forest": "#0F766E", "XGBoost": "#B56B08"}
    for name, curve in curves.items():
        ax.plot(curve["fpr"], curve["tpr"], color=colors.get(name), label=f"{name} (AUC {evaluations[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8492A6", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Final test ROC · model selection already frozen")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_cv_plot(comparison: pd.DataFrame, path) -> None:
    available = comparison.loc[comparison["status"] == "trained"].sort_values("cv_roc_auc_mean")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(available["model"], available["cv_roc_auc_mean"], xerr=available["cv_roc_auc_std"], color="#2463EB", capsize=4)
    ax.set(xlim=(0, 1.02), xlabel="Training CV ROC-AUC (mean ± fold SD)", title="Model comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
