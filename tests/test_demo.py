import pytest

from heart_disease.config import FEATURES, UCI_URL
from heart_disease.demo import public_demo_record, result_export


def test_manual_source_never_presented_as_public(dataset):
    with pytest.raises(ValueError, match="official UCI"):
        public_demo_record({"data": dataset.metadata})


def test_public_example_hash_and_export_without_measurements(dataset):
    # A mocked official source in an isolated fixture; not actual public data.
    data = {**dataset.metadata, "source": {"url": UCI_URL}}
    metadata = {"data": data, "run_id": "unit-only", "selected_model": "unit classifier", "probability_method": "raw"}
    record, label = public_demo_record(metadata)
    assert set(record) == set(FEATURES) and "Functional demonstration" in label
    result = result_export({"prediction": 1, "positive_class_probability": 0.7}, metadata, example_loaded=True)
    assert set(result).isdisjoint(FEATURES)
    assert result["public_example_loaded"] is True
    assert result["predicted_dataset_class"] == 1
    metadata["data"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        public_demo_record(metadata)
