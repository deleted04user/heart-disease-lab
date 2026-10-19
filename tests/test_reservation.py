import pytest

from heart_disease.reservation import ensure_not_reserved, reserve_cohort


def test_reserved_bytes_are_not_parsed_and_cannot_be_loaded(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Predeclared unit test protocol; fixture bytes have no outcomes.")
    source, manifest = tmp_path / "opaque.data", tmp_path / "reservation.json"
    monkeypatch.setattr(
        "heart_disease.data._fetch", lambda _: b"opaque fixture bytes, not dataset records"
    )
    result = reserve_cohort(protocol, source, manifest)
    assert result["labels_inspected"] is False and result["model_evaluated"] is False
    assert reserve_cohort(protocol, source, manifest) == result
    with pytest.raises(ValueError, match="reserved cohort"):
        ensure_not_reserved(source, manifest)
    copied = tmp_path / "renamed.csv"
    copied.write_bytes(source.read_bytes())
    with pytest.raises(ValueError):
        ensure_not_reserved(copied, manifest)
    protocol.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        reserve_cohort(protocol, source, manifest)
