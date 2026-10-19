"""Load trusted local artifacts once and validate all reusable inference inputs."""
import importlib.metadata
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.utils.validation import check_is_fitted

from .config import FEATURES, MODEL_PATH
from .data import validate_features
from .io import read_json, sha256, write_json


class ArtifactError(RuntimeError):
    """Missing, damaged, or incompatible model artifact."""


def metadata_path(path: Path) -> Path:
    return Path(path).with_suffix(".metadata.json")


def save_artifact(pipeline, metadata: dict, path: Path = MODEL_PATH) -> dict:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    check_is_fitted(pipeline.named_steps["classifier"])
    temporary = path.with_suffix(".tmp")
    joblib.dump(pipeline, temporary, compress=3)
    temporary.replace(path)
    complete = {**metadata, "artifact_schema": 1, "model_sha256": sha256(path), "features": FEATURES}
    write_json(metadata_path(path), complete)
    return complete


def load_artifact(path: Path = MODEL_PATH):
    """Only load artifacts you created/trust: joblib can execute Python code."""
    path = Path(path).resolve()
    if not path.is_file() or not metadata_path(path).is_file():
        raise ArtifactError(f"Model or metadata missing at {path}. Run python scripts/train.py first.")
    try:
        metadata = read_json(metadata_path(path))
        if metadata.get("artifact_schema") != 1 or metadata.get("features") != FEATURES:
            raise ValueError("Unsupported artifact version or feature schema.")
        if metadata.get("model_sha256") != sha256(path):
            raise ValueError("Model checksum differs from its metadata. Recreate the artifact.")
        for package in ["scikit-learn", "numpy", "pandas", "joblib"] + (["xgboost"] if metadata.get("selected_model") == "XGBoost" else []):
            trained = metadata.get("dependency_versions", {}).get(package)
            current = importlib.metadata.version(package)
            if trained and trained.split(".")[:2] != current.split(".")[:2]:
                raise ValueError(f"{package} version mismatch: trained {trained}, installed {current}. Use requirements-lock.txt or retrain.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = joblib.load(path)
        check_is_fitted(pipeline.named_steps["classifier"])
        if list(pipeline.feature_names_in_) != FEATURES or set(pipeline.classes_) != {0, 1}:
            raise ValueError("Pipeline feature/class schema is incompatible.")
    except Exception as exc:
        raise ArtifactError(f"Cannot load trusted model at {path}: {exc}. Retrain with this environment.") from exc
    metadata["_artifact_dir"] = str(path.parent)
    return pipeline, metadata


def predict_records(pipeline, records: dict | list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame([records] if isinstance(records, dict) else records)
    X = validate_features(frame, allow_missing=False)
    result = pd.DataFrame({"prediction": pipeline.predict(X).astype(int)})
    if hasattr(pipeline, "predict_proba"):
        positive_index = int(np.flatnonzero(pipeline.classes_ == 1)[0])
        result["positive_class_probability"] = pipeline.predict_proba(X)[:, positive_index]
    return result
