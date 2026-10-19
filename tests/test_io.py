"""Interrupted writes must preserve the last complete JSON document."""

from concurrent.futures import ThreadPoolExecutor

import pytest

import heart_disease.io as storage


@pytest.mark.parametrize("invalid", [{"value": float("nan")}, {"value": {1, 2}}])
def test_serialization_failure_preserves_previous_document(tmp_path, invalid):
    path = tmp_path / "report.json"
    storage.write_json(path, {"run_id": "original"})
    original = path.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        storage.write_json(path, invalid)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_failed_write_preserves_previous_document_and_cleans_staging(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "report.json"
    storage.write_json(path, {"run_id": "original"})
    original = path.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError("isolated write failure")

    monkeypatch.setattr(storage.os, failure, interrupted)
    with pytest.raises(OSError, match="isolated write failure"):
        storage.write_json(path, {"run_id": "replacement"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_writes_use_separate_staging_files(tmp_path):
    path = tmp_path / "nested/report.json"
    values = [{"writer": index, "payload": "é" * 1000} for index in range(12)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda value: storage.write_json(path, value), values))
    assert storage.read_json(path) in values
    assert list(path.parent.iterdir()) == [path]
