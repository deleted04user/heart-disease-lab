"""Strict loaders for original UCI codes and the documented Kaggle UCI mirror."""

import csv
import hashlib
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .config import (
    BOUNDS,
    CATEGORY_LABELS,
    DATA_PATH,
    FEATURES,
    TARGET,
    UCI_ARCHIVE,
    UCI_PAGE,
    UCI_URL,
)
from .io import read_json, sha256, write_json

LOG = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """The data cannot safely be interpreted as the documented schema."""


@dataclass
class Dataset:
    frame: pd.DataFrame
    metadata: dict


def validate_features(frame: pd.DataFrame, *, allow_missing: bool) -> pd.DataFrame:
    """Validate canonical input codes and return numbers in canonical order."""
    if frame.columns.duplicated().any():
        raise DataValidationError("Duplicate feature column names are not allowed.")
    missing = set(FEATURES) - set(frame.columns)
    extra = set(frame.columns) - set(FEATURES)
    if missing or extra:
        raise DataValidationError(
            f"Feature schema mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    if frame.empty:
        raise DataValidationError("No feature records supplied.")
    output = frame[FEATURES].copy()
    output = output.mask(output.isin(["?", "", "NA", "N/A", "NaN", "nan", "null"]))
    for col in FEATURES:
        try:
            output[col] = pd.to_numeric(output[col], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(
                f"{col}: expected numeric values in canonical UCI codes."
            ) from exc
        values = output[col].dropna()
        if not np.isfinite(values).all():
            raise DataValidationError(f"{col}: non-finite numeric values are invalid.")
        if not allow_missing and output[col].isna().any():
            raise DataValidationError(f"{col}: missing values are not accepted for prediction.")
        if col in CATEGORY_LABELS:
            invalid = sorted(set(values) - set(CATEGORY_LABELS[col]))
            if invalid:
                raise DataValidationError(
                    f"{col}: invalid codes {invalid}; expected {list(CATEGORY_LABELS[col])}. Check the source encoding."
                )
        else:
            low, high = BOUNDS[col]
            if not values.between(low, high).all():
                raise DataValidationError(
                    f"{col}: values must be between {low} and {high}. Inspect zero placeholders and source units; nothing was corrected automatically."
                )
            if col != "oldpeak" and (values % 1 != 0).any():
                raise DataValidationError(f"{col}: expected whole-number measurements.")
    return output


def normalize_target(series: pd.Series, semantics: str) -> pd.Series:
    """Transform only explicitly asserted UCI 0–4 or positive-is-presence binary labels."""
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Target must contain numeric diagnosis labels.") from exc
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise DataValidationError("Missing or non-finite target labels are not allowed.")
    allowed = {0, 1, 2, 3, 4} if semantics == "uci" else {0, 1} if semantics == "binary" else None
    if allowed is None:
        raise DataValidationError(
            "Specify target semantics: 'uci' (0–4) or 'binary' (1 means presence)."
        )
    if not set(numeric).issubset(allowed):
        raise DataValidationError(
            f"Invalid target values {sorted(set(numeric))}; expected {sorted(allowed)}."
        )
    result = (numeric > 0).astype(int).rename(TARGET)
    if result.nunique() != 2:
        raise DataValidationError("Both target classes must be present.")
    return result


def _parse_csv(path: Path) -> tuple[pd.DataFrame, bool, int]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle, strict=True))
    except (csv.Error, UnicodeError) as exc:
        raise DataValidationError(f"Malformed CSV: {exc}") from exc
    if not rows or not rows[0]:
        raise DataValidationError("Dataset file is empty.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise DataValidationError(
            "Malformed rows: inconsistent column counts or blank lines. No rows were skipped."
        )
    first = [cell.strip() for cell in rows[0]]
    # A header is identified by known names, never merely by non-numeric contents.
    header = any(cell.lower() in {"age", "num", "target", "heartdisease"} for cell in first)
    if header:
        names = [cell.lower() for cell in first]
        if len(set(names)) != len(names):
            raise DataValidationError("Duplicate column names in CSV header.")
        if len(rows) == 1:
            raise DataValidationError("Dataset has a header but no records.")
        return pd.DataFrame(rows[1:], columns=names), False, width
    if width != 14:
        raise DataValidationError(
            f"Headerless Cleveland data must have 14 columns; received {width}."
        )
    return pd.DataFrame(rows, columns=FEATURES + ["num"]), True, width


def _decode_mirror(frame: pd.DataFrame) -> pd.DataFrame:
    """Mappings from redwankarimsony/heart-disease-data (not recoded heart.csv)."""
    mappings = {
        "sex": {"male": 1, "female": 0},
        "cp": {"typical angina": 1, "atypical angina": 2, "non-anginal": 3, "asymptomatic": 4},
        "fbs": {"true": 1, "false": 0},
        "restecg": {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2},
        "exang": {"true": 1, "false": 0},
        "slope": {"upsloping": 1, "flat": 2, "downsloping": 3},
        "thal": {"normal": 3, "fixed defect": 6, "reversable defect": 7, "reversible defect": 7},
    }
    output = frame.copy()
    for col, mapping in mappings.items():
        if col not in output:
            continue  # Missing-column validation gives the complete schema error below.
        values = output[col].astype("string").str.strip().str.lower()
        unknown = set(values.dropna()) - set(mapping)
        if unknown:
            raise DataValidationError(
                f"Kaggle {col}: unrecognized labels {sorted(unknown)}. This loader expects the documented string-coded UCI mirror."
            )
        output[col] = values.map(mapping)
    return output


def load_dataset(
    path: Path = DATA_PATH,
    *,
    schema: str = "uci",
    target_semantics: str | None = None,
    site: str = "Cleveland",
) -> Dataset:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run python scripts/download_data.py or supply --data PATH. See README manual download instructions."
        )
    from .reservation import ensure_not_reserved

    ensure_not_reserved(path)
    if schema not in {"uci", "kaggle"}:
        raise DataValidationError("Supported schemas: uci, kaggle.")
    frame, headerless, raw_columns = _parse_csv(path)
    input_rows = len(frame)
    frame = frame.apply(lambda col: col.str.strip())
    frame = frame.mask(frame.isin(["?", "", "NA", "N/A", "NaN", "nan", "null"]))
    notes = []
    excluded = 0
    if schema == "kaggle":
        if headerless or "dataset" not in frame:
            raise DataValidationError(
                "Kaggle mirror requires a named header and a dataset column so sites cannot be silently mixed."
            )
        sites = sorted(frame["dataset"].dropna().unique().tolist())
        if frame["dataset"].isna().any() or site not in sites:
            raise DataValidationError(
                f"Choose a valid --site from {sites}; dataset values cannot be missing."
            )
        frame = frame.loc[frame["dataset"] == site].copy()
        excluded = input_rows - len(frame)
        notes.append(f"Explicit site selection '{site}' excluded {excluded} rows from other sites.")
        frame = frame.drop(columns=[col for col in ["id", "dataset"] if col in frame])
        if "thalch" in frame:
            if "thalach" in frame:
                raise DataValidationError(
                    "Both thalch and thalach are present; mapping is ambiguous."
                )
            frame = frame.rename(columns={"thalch": "thalach"})
        frame = _decode_mirror(frame)
    targets = [col for col in ["num", "target", "heartdisease"] if col in frame]
    if len(targets) != 1:
        raise DataValidationError(
            "Expected exactly one target column: num, target, or HeartDisease. Multiple targets risk leakage."
        )
    original_target = targets[0]
    semantics = target_semantics or ("uci" if original_target == "num" else None)
    if semantics is None:
        raise DataValidationError(
            "Target meaning is ambiguous. Pass --target-semantics binary only after verifying 0=absence and 1=presence, or uci for original 0–4 diagnosis labels."
        )
    target_counts = frame[original_target].value_counts().to_dict()
    y = normalize_target(frame.pop(original_target), semantics)
    X = validate_features(frame, allow_missing=True)
    if X.isna().all().any():
        raise DataValidationError(
            f"Entirely missing features cannot be trained: {X.columns[X.isna().all()].tolist()}"
        )
    normalized = X.assign(target=y)
    selected_rows = len(normalized)
    duplicate_count = int(normalized.duplicated().sum())
    # Identical observations must never straddle training and test partitions.
    normalized = normalized.drop_duplicates().reset_index(drop=True)
    if normalized.duplicated(subset=FEATURES, keep=False).any():
        raise DataValidationError(
            "Identical feature records have conflicting labels. Resolve their provenance before training; no records were silently discarded."
        )
    if duplicate_count:
        notes.append(
            f"Removed {duplicate_count} exact duplicate normalized records before splitting to prevent leakage."
        )
    if int(normalized[TARGET].value_counts().min()) < 3:
        raise DataValidationError(
            "Each class needs at least 3 distinct records for a held-out test and stratified training CV."
        )
    source = {"source": f"Manual local file: {path.name}", "url": None}
    sidecar = path.with_suffix(path.suffix + ".source.json")
    if sidecar.exists():
        provenance = read_json(sidecar)
        if provenance.get("sha256") == sha256(path):
            source = provenance
        else:
            notes.append(
                "Download provenance checksum differs: treating file as a modified local source."
            )
    verified = source.get("url") in {UCI_URL, UCI_ARCHIVE} and selected_rows == 303 and headerless
    variant = (
        "UCI original processed Cleveland (303 records verified)"
        if verified
        else f"{'Kaggle UCI mirror, site ' + site if schema == 'kaggle' else 'Local UCI-coded 14-column dataset'}; source identity not independently verified"
    )
    if (
        source.get("member") == "processed.hungarian.data"
        and source.get("url")
        in {
            UCI_ARCHIVE,
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        }
        and selected_rows == 294
        and headerless
    ):
        variant = "UCI original processed Hungarian (294 source rows verified)"
    for note in notes:
        LOG.warning(note)
    metadata = {
        "source": source,
        "reference_url": UCI_PAGE,
        "variant": variant,
        "schema": schema,
        "site": site if schema == "kaggle" else None,
        "path": str(path),
        "sha256": sha256(path),
        "raw_rows": input_rows,
        "raw_columns": raw_columns,
        "selected_rows": selected_rows,
        "excluded_other_sites": excluded,
        "rows": len(normalized),
        "columns": len(normalized.columns),
        "feature_count": len(FEATURES),
        "duplicates_removed": duplicate_count,
        "original_target_column": original_target,
        "original_target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_semantics": semantics,
        "target_mapping": "0 → 0; 1,2,3,4 → 1"
        if semantics == "uci"
        else "0 → 0; 1 → 1 (user asserted positive = disease presence)",
        "target_distribution": {
            str(k): int(v) for k, v in normalized[TARGET].value_counts().sort_index().items()
        },
        "missing_values": {col: int(v) for col, v in normalized.isna().sum().items()},
        "observed_ranges": {
            col: {"min": float(normalized[col].min()), "max": float(normalized[col].max())}
            for col in FEATURES
        },
        "notes": notes,
        "records_corrected": 0,
    }
    return Dataset(normalized, metadata)


def _fetch(url: str) -> bytes:
    request = Request(
        url, headers={"User-Agent": "HeartDiseaseLab/1.0 (educational dataset retrieval)"}
    )
    with urlopen(request, timeout=30) as response:
        data = response.read(5_000_001)
    if len(data) > 5_000_000:
        raise DataValidationError("Unexpectedly large download; refused to write it.")
    return data


def download_dataset(path: Path = DATA_PATH) -> Dataset:
    """Reuse existing data. Otherwise try original file, then official archive."""
    path = Path(path).resolve()
    if path.exists():
        LOG.info("Existing data preserved and validated: %s", path)
        return load_dataset(path)
    errors = []
    for url in [UCI_URL, UCI_ARCHIVE]:
        try:
            payload = _fetch(url)
            if url == UCI_ARCHIVE:
                with ZipFile(io.BytesIO(payload)) as archive:
                    matches = [
                        name
                        for name in archive.namelist()
                        if Path(name).name == "processed.cleveland.data"
                    ]
                    if len(matches) != 1:
                        raise DataValidationError(
                            "Official archive did not contain one Cleveland file."
                        )
                    if archive.getinfo(matches[0]).file_size > 5_000_000:
                        raise DataValidationError("Cleveland archive entry is unexpectedly large.")
                    payload = archive.read(matches[0])
            # Validate in a separate temporary file before creating the destination.
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                candidate = Path(tmp) / "processed.cleveland.data"
                candidate.write_bytes(payload)
                checked = load_dataset(candidate)
                if checked.metadata["selected_rows"] != 303:
                    raise DataValidationError(
                        "Official Cleveland file did not have the expected 303 records."
                    )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
            write_json(
                path.with_suffix(path.suffix + ".source.json"),
                {
                    "source": "UCI Machine Learning Repository",
                    "url": url,
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            )
            return load_dataset(path)
        except (URLError, TimeoutError, OSError, BadZipFile, DataValidationError) as exc:
            if path.exists():
                raise RuntimeError(
                    f"Destination now exists; it was not overwritten: {path}. {exc}"
                ) from exc
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Official download failed. Manually download processed.cleveland.data from "
        + UCI_PAGE
        + f" and place it at {path}. No synthetic data was substituted.\n"
        + "\n".join(errors)
    )
