import numpy as np
import pandas as pd
import pytest

from heart_disease.config import FEATURES
from heart_disease.data import DataValidationError
from heart_disease.io import write_json
from heart_disease.modeling import build_models, dependency_versions
from heart_disease.prediction import ArtifactError, load_artifact, metadata_path, predict_records, save_artifact


@pytest.fixture
def fitted(dataset):
    pipeline = build_models()[0]["Logistic Regression"]
    return pipeline.fit(dataset.frame[FEATURES], dataset.frame.target)


def test_output_shape_and_save_reload_consistency(fitted, dataset, tmp_path):
    path = tmp_path / "unit_fixture_model.joblib"
    metadata = {"selected_model": "Logistic Regression", "dependency_versions": dependency_versions()}
    save_artifact(fitted, metadata, path)
    restored, stored = load_artifact(path)
    before = predict_records(fitted, dataset.frame[FEATURES].iloc[:3])
    after = predict_records(restored, dataset.frame[FEATURES].iloc[:3])
    pd.testing.assert_frame_equal(before, after)
    assert before.shape == (3, 2)
    assert before.prediction.dtype.kind in "iu"
    assert before.positive_class_probability.between(0, 1).all()
    assert stored["features"] == FEATURES
    assert predict_records(restored, dataset.frame[FEATURES].iloc[0].to_dict()).shape == (1, 2)


def test_prediction_rejects_missing_extra_unknown_and_ranges(fitted, dataset):
    valid = dataset.frame[FEATURES].iloc[:1].copy()
    cases = [valid.drop(columns="age"), valid.assign(target=0), valid.assign(ca=99), valid.assign(chol=0), valid.assign(age=np.nan), valid.iloc[:0]]
    for invalid in cases:
        with pytest.raises(DataValidationError):
            predict_records(fitted, invalid)


def test_missing_corrupt_incompatible_artifacts(fitted, tmp_path):
    path = tmp_path / "fixture.joblib"
    with pytest.raises(ArtifactError, match="missing"):
        load_artifact(path)
    save_artifact(fitted, {"dependency_versions": {"scikit-learn": "0.0.1"}}, path)
    with pytest.raises(ArtifactError, match="version mismatch"):
        load_artifact(path)
    save_artifact(fitted, {}, path)
    path.write_bytes(b"corrupted")
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifact(path)
    write_json(metadata_path(path), {"artifact_schema": 100})
    with pytest.raises(ArtifactError, match="Unsupported"):
        load_artifact(path)
