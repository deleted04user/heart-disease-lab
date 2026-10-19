from urllib.error import URLError

import pytest

from heart_disease.config import FEATURES
from heart_disease.data import DataValidationError
from heart_disease.external import download_external, evaluate_external
from heart_disease.modeling import build_models


def test_external_pipeline_never_fits_and_overlap_is_audited(dataset, frame, tmp_path, monkeypatch):
    pipeline = build_models()[0]["Logistic Regression"].fit(
        dataset.frame[FEATURES], dataset.frame.target
    )
    external = frame.copy()
    external.loc[1:, "chol"] += 1
    path = tmp_path / "unit_external.csv"
    external.to_csv(path, index=False)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("External evaluation must not fit or calibrate on the external cohort")

    monkeypatch.setattr(pipeline, "fit", forbidden_fit)
    result = evaluate_external(pipeline, dataset, path, resamples=100)
    assert result["fitting_on_external_data"] is False
    assert result["rows_evaluated"] == len(frame) - 1
    assert result["exclusions"]["feature_overlap_with_cleveland"] == 1
    assert result["overlapping_normalized_row_indices_excluded"] == [0]
    with pytest.raises(DataValidationError, match="training source"):
        evaluate_external(pipeline, dataset, dataset.metadata["path"], resamples=100)


def test_existing_external_preserved_without_network(frame, tmp_path, monkeypatch):
    path = tmp_path / "unit_fixture.data"
    frame.to_csv(path, index=False, header=False)
    original = path.read_bytes()
    monkeypatch.setattr(
        "heart_disease.external._fetch", lambda _: pytest.fail("Existing data must be preserved")
    )
    assert len(download_external(path).frame) == len(frame)
    assert path.read_bytes() == original


def test_external_download_failures_are_explicit_and_write_no_data(tmp_path, monkeypatch):
    path = tmp_path / "missing.data"

    def blocked(url):
        if url.endswith(".data"):
            raise URLError("isolated unit network failure")
        return b"invalid zip fixture"

    monkeypatch.setattr("heart_disease.external._fetch", blocked)
    with pytest.raises(RuntimeError, match="No substitute"):
        download_external(path)
    assert not path.exists()
