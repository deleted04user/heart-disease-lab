"""Audited evaluation on a separate official UCI site; never fit on it."""
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .config import FEATURES, ROOT, UCI_ARCHIVE
from .data import Dataset, DataValidationError, _fetch, load_dataset
from .evaluation import evaluate
from .io import sha256, write_json
from .research import probability_scores, reliability_bins, subgroup_results, uncertainty

EXTERNAL_PATH = ROOT / "data/raw/processed.hungarian.data"
EXTERNAL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"


def download_external(path: Path = EXTERNAL_PATH) -> Dataset:
    path = Path(path).resolve()
    if path.exists():
        return load_dataset(path)
    errors = []
    for url in [EXTERNAL_URL, UCI_ARCHIVE]:
        try:
            content = _fetch(url)
            if url == UCI_ARCHIVE:
                with ZipFile(io.BytesIO(content)) as archive:
                    names = [name for name in archive.namelist() if Path(name).name == "processed.hungarian.data"]
                    if len(names) != 1 or archive.getinfo(names[0]).file_size > 5_000_000:
                        raise DataValidationError("Unexpected Hungarian archive member.")
                    content = archive.read(names[0])
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                candidate = Path(tmp) / "hungarian.data"
                candidate.write_bytes(content)
                data = load_dataset(candidate)
                if data.metadata["selected_rows"] != 294:
                    raise DataValidationError("Expected 294 original Hungarian source rows.")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(content)
            write_json(path.with_suffix(path.suffix + ".source.json"), {"source": "UCI Machine Learning Repository · processed Hungarian cohort", "url": url, "member": "processed.hungarian.data", "retrieved_utc": datetime.now(timezone.utc).isoformat(), "sha256": hashlib.sha256(content).hexdigest()})
            return load_dataset(path)
        except (OSError, ValueError, TimeoutError, BadZipFile) as exc:
            if path.exists():
                raise RuntimeError(f"Existing destination preserved: {path}. {exc}") from exc
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"External download unavailable. Manually place processed.hungarian.data at {path}. No substitute was used.\n" + "\n".join(errors))


def evaluate_external(pipeline, reference: Dataset, path: Path, *, resamples: int = 1000) -> dict:
    dataset = load_dataset(path)
    if dataset.metadata["sha256"] == reference.metadata["sha256"]:
        raise DataValidationError("External source is the training source; independent-site evaluation refused.")
    # Conservative overlap audit against ALL Cleveland observations, including
    # holdout, before evaluating. Missing values are equal for this comparison.
    reference_keys = set(pd.util.hash_pandas_object(reference.frame[FEATURES], index=False).tolist())
    keys = pd.util.hash_pandas_object(dataset.frame[FEATURES], index=False)
    overlap = keys.isin(reference_keys)
    excluded_rows = np.flatnonzero(overlap).tolist()
    frame = dataset.frame.loc[~overlap].copy()
    if frame.empty or frame.target.nunique() != 2:
        raise DataValidationError("No valid two-class external cohort remains after overlap auditing.")
    X, y = frame[FEATURES], frame.target
    probabilities = pipeline.predict_proba(X)[:, 1]
    predicted = pipeline.predict(X)
    missing = {col: {"count": int(X[col].isna().sum()), "fraction": float(X[col].isna().mean())} for col in FEATURES}
    outside = {col: int(((X[col] < reference.metadata["observed_ranges"][col]["min"]) | (X[col] > reference.metadata["observed_ranges"][col]["max"])).sum()) for col in FEATURES}
    source = dataset.metadata
    return {
        "status": "evaluated", "cohort": "UCI processed Hungarian" if source["source"].get("member") == "processed.hungarian.data" or Path(path).name == "processed.hungarian.data" else "Explicit local external cohort (identity asserted by caller)",
        "data": source, "rows_evaluated": len(frame), "overlapping_normalized_row_indices_excluded": excluded_rows,
        "exclusions": {"exact_duplicates": source["duplicates_removed"], "feature_overlap_with_cleveland": len(excluded_rows), "invalid_rows_silently_removed": 0},
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "missingness": missing, "outside_cleveland_observed_ranges": outside,
        "metrics": evaluate(y, predicted, probabilities), "probability_scores": probability_scores(y, probabilities),
        "uncertainty": uncertainty(y, predicted, probabilities, resamples=resamples),
        "subgroups": subgroup_results(X, y, predicted, probabilities),
        "reliability": reliability_bins(y, probabilities),
        "fitting_on_external_data": False,
        "caution": "Historical site-transfer check, not prospective clinical validation. Severe feature missingness can make Cleveland-trained imputation unsuitable. Original UCI diagnosis semantics are asserted and validated; input measurement bounds are enforced. No invalid rows are silently corrected or skipped. Feature-overlap audit cannot rule out shared patient identities. Do not tune using this cohort after viewing results.",
    }
