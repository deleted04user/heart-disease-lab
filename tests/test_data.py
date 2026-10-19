from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from heart_disease.config import CATEGORY_LABELS, FEATURES
from heart_disease.data import DataValidationError, download_dataset, load_dataset, normalize_target, validate_features


def test_headerless_parsing_and_missing(frame, tmp_path):
    frame.loc[0, "ca"] = np.nan
    frame.loc[1, "thal"] = np.nan
    path = tmp_path / "cleveland.data"
    frame.to_csv(path, header=False, index=False, na_rep="?")
    loaded = load_dataset(path)
    assert len(loaded.frame) == len(frame)
    assert loaded.metadata["raw_columns"] == 14
    assert loaded.metadata["missing_values"]["ca"] == 1
    assert list(loaded.frame.columns) == FEATURES + ["target"]
    assert "303 records verified" not in loaded.metadata["variant"]


@pytest.mark.parametrize("target,semantics,expected", [([0, 1, 2, 3, 4], "uci", [0, 1, 1, 1, 1]), ([0, 1, 1], "binary", [0, 1, 1])])
def test_target_normalization(target, semantics, expected):
    assert normalize_target(pd.Series(target), semantics).tolist() == expected


@pytest.mark.parametrize("target,semantics", [([0, 5], "uci"), ([0, -1], "uci"), ([0, 0.5], "uci"), ([0, 2], "binary"), ([0, np.nan], "uci"), ([0, np.inf], "uci"), ([0, "yes"], "uci"), ([0, 0], "uci"), ([0, 1], "unknown")])
def test_invalid_targets(target, semantics):
    with pytest.raises(DataValidationError):
        normalize_target(pd.Series(target), semantics)


def test_binary_header_requires_semantic_assertion(frame, tmp_path):
    path = tmp_path / "binary.csv"
    frame.rename(columns={"num": "HeartDisease"}).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="ambiguous"):
        load_dataset(path)
    assert set(load_dataset(path, target_semantics="binary").frame.target) == {0, 1}


@pytest.mark.parametrize("text", ["", "age,age,num\n50,1,0\n", "1,2,3\n", "age,num\n50,0\n51,1,4\n", "age,num\n", '"unterminated\n'])
def test_empty_malformed_duplicate_columns(text, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(DataValidationError):
        load_dataset(path)


def test_missing_features_extra_target_and_all_missing(frame, tmp_path):
    path = tmp_path / "bad.csv"
    for changed in [frame.drop(columns="age"), frame.assign(target=frame.num), frame.assign(ca=np.nan), frame.assign(secret_target=frame.num)]:
        changed.to_csv(path, index=False)
        with pytest.raises(DataValidationError):
            load_dataset(path)


@pytest.mark.parametrize("col,value", [("age", "oops"), ("age", 151), ("age", 50.5), ("chol", 0), ("cp", 0), ("thal", 2), ("ca", 4), ("oldpeak", np.inf)])
def test_invalid_feature_values(frame, col, value):
    X = frame[FEATURES].astype(object)
    X.loc[0, col] = value
    with pytest.raises(DataValidationError):
        validate_features(X, allow_missing=True)


def test_duplicates_removed_and_conflicting_labels_rejected(frame, tmp_path):
    path = tmp_path / "duplicates.csv"
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded.metadata["duplicates_removed"] == 1
    assert len(loaded.frame) == len(frame)
    assert loaded.metadata["notes"]
    changed = frame.iloc[[0]].copy()
    changed["num"] = 1
    pd.concat([frame, changed]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="conflicting"):
        load_dataset(path)


def test_insufficient_classes(frame, tmp_path):
    path = tmp_path / "small.csv"
    frame.iloc[:4].to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="at least 3"):
        load_dataset(path)


def test_kaggle_mirror_explicit_mapping_and_site_selection(frame, tmp_path):
    mirror = frame.copy()
    mirror["dataset"] = "Cleveland"
    mirror["id"] = np.arange(len(frame))
    mapping = {"sex": {0: "Female", 1: "Male"}, "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"}, "fbs": {0: "False", 1: "True"}, "restecg": {0: "normal", 1: "st-t abnormality", 2: "lv hypertrophy"}, "exang": {0: "False", 1: "True"}, "slope": {1: "upsloping", 2: "flat", 3: "downsloping"}, "thal": {3: "normal", 6: "fixed defect", 7: "reversable defect"}}
    for col, labels in mapping.items():
        mirror[col] = mirror[col].map(labels)
    mirror = mirror.rename(columns={"thalach": "thalch"})
    others = mirror.iloc[:4].assign(dataset="Hungary", chol=0)
    path = tmp_path / "mirror.csv"
    pd.concat([mirror, others]).to_csv(path, index=False)
    loaded = load_dataset(path, schema="kaggle")
    np.testing.assert_allclose(loaded.frame[FEATURES], frame[FEATURES])
    assert loaded.metadata["excluded_other_sites"] == 4
    with pytest.raises(DataValidationError, match="valid --site"):
        load_dataset(path, schema="kaggle", site="Missing")
    mirror.loc[0, "sex"] = "unrecognized"
    mirror.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="unrecognized labels"):
        load_dataset(path, schema="kaggle")


def test_existing_download_preserved(frame, tmp_path, monkeypatch):
    path = tmp_path / "local.data"
    frame.to_csv(path, header=False, index=False)
    before = path.read_bytes()
    def fail(_):
        pytest.fail("Existing data must never cause a network request")
    monkeypatch.setattr("heart_disease.data._fetch", fail)
    download_dataset(path)
    assert path.read_bytes() == before


def test_download_failure_clear_and_no_fallback_data(tmp_path, monkeypatch):
    def blocked(_):
        raise URLError("network blocked for unit test")
    monkeypatch.setattr("heart_disease.data._fetch", blocked)
    path = tmp_path / "absent.data"
    with pytest.raises(RuntimeError, match="Manually download"):
        download_dataset(path)
    assert not path.exists()


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_dataset(tmp_path / "missing.csv")
