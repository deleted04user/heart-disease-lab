"""Held-out permutation importance at the original input feature level."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import FEATURES, SEED


def explain_model(pipeline, X_test, y_test, output_dir, *, repeats: int = 20) -> pd.DataFrame:
    result = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc", n_repeats=repeats, random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({"feature": FEATURES, "importance_mean": result.importances_mean, "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)
    ordered = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"], color="#2463EB", capsize=3)
    ax.axvline(0, color="#8492A6", linewidth=0.8)
    ax.set(xlabel="Decrease in held-out ROC-AUC (mean ± repeat SD)", title="Which inputs does this model use?")
    fig.tight_layout()
    fig.savefig(output_dir / "figures/permutation_importance.png", dpi=160)
    plt.close(fig)
    # Native coefficients/importances are supplementary transformed-column data;
    # they are never mislabeled as original-feature or causal importance.
    classifier = pipeline.named_steps["classifier"]
    if "preprocess" not in pipeline.named_steps:
        # A calibrated ensemble has several independently fitted preprocessors;
        # a single coefficient table would misrepresent that fitted estimator.
        (output_dir / "transformed_weights_note.txt").write_text("Calibrated ensemble: no single transformed coefficient vector. Use original-feature permutation importance.\n", encoding="utf-8")
        return importance
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values, label = classifier.coef_[0], "coefficient_log_odds"
    elif hasattr(classifier, "feature_importances_"):
        values, label = classifier.feature_importances_, "native_importance"
    else:
        return importance
    pd.DataFrame({"transformed_feature": names, label: values}).to_csv(output_dir / "transformed_model_weights.csv", index=False)
    return importance
