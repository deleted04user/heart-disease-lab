"""Public-example loading and minimal result exports, with no input persistence."""
from pathlib import Path

from .config import DISCLAIMER, FEATURES
from .data import load_dataset
from .data import validate_features
from .io import read_json, sha256
import pandas as pd


def public_demo_record(metadata: dict, example_index: int = 0) -> tuple[dict, str]:
    source = metadata["data"]
    url = source.get("source", {}).get("url") or ""
    if not url.startswith("https://archive.ics.uci.edu/"):
        raise ValueError("The demo loader is available only for a verified official UCI source; private/manual records are not treated as public examples.")
    if metadata.get("public_demo"):
        configuration = metadata["public_demo"]
        base = Path(metadata["_artifact_dir"]).resolve()
        path = (base / configuration["path"]).resolve()
        if not path.is_relative_to(base) or sha256(path) != configuration["sha256"]:
            raise ValueError("Bundled public examples failed their path/checksum validation.")
        examples = read_json(path)
        if examples.get("schema") != 1 or examples.get("source_sha256") != source["sha256"] or examples.get("source_url") != url:
            raise ValueError("Bundled example provenance differs from model metadata.")
        if not 0 <= example_index < len(examples["records"]):
            raise ValueError("Choose a valid public example index.")
        observation = examples["records"][example_index]
        row = validate_features(pd.DataFrame([observation["features"]]), allow_missing=False).iloc[0]
        record = {col: float(row[col]) if col == "oldpeak" else int(row[col]) for col in FEATURES}
        return record, f"Public UCI example · normalized source row {observation['source_row']}. Functional demonstration only; this is not an independent evaluation."
    dataset = load_dataset(Path(source["path"]), schema=source["schema"], target_semantics=source["target_semantics"], site=source.get("site") or "Cleveland")
    if dataset.metadata["sha256"] != source["sha256"]:
        raise ValueError("Source changed since training; the public example was not loaded.")
    complete = dataset.frame[FEATURES].dropna()
    if complete.empty:
        raise ValueError("No complete public observation is available for the 13-field form.")
    row = complete.iloc[0]
    record = {col: float(row[col]) if col == "oldpeak" else int(row[col]) for col in FEATURES}
    return record, f"Public UCI example · normalized source row {int(row.name) + 1}. Functional demonstration only; this is not an independent evaluation."


def result_export(result: dict, metadata: dict, *, example_loaded: bool) -> dict:
    """Export model output and provenance; deliberately omit input measurements."""
    return {
        "run_id": metadata["run_id"], "model": metadata["selected_model"],
        "probability_method": metadata.get("probability_method", "raw"),
        "predicted_dataset_class": int(result["prediction"]),
        "model_estimated_positive_class_probability": float(result["positive_class_probability"]) if "positive_class_probability" in result else None,
        "threshold": 0.5, "public_example_loaded": bool(example_loaded),
        "interpretation": "Dataset label only; not a confirmed diagnosis or clinically validated risk score.",
        "disclaimer": DISCLAIMER,
    }
