"""Recoverable dashboard failures and report identity boundaries."""

from heart_disease.ui import common


def test_json_reader_recovers_from_invalid_or_missing_reports(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(common.st, "warning", warnings.append)
    path = tmp_path / "report.json"
    assert common.json_file(path) is None
    path.write_text("[]")
    assert common.json_file(path) is None
    path.write_text("broken")
    assert common.json_file(path) is None
    path.write_text('{"run_id": "one"}')
    assert common.json_file(path) == {"run_id": "one"}
    assert len(warnings) == 2


def test_report_from_another_model_is_rejected(monkeypatch):
    errors, recovery = [], []
    monkeypatch.setattr(common.st, "error", errors.append)
    monkeypatch.setattr(common, "recover", recovery.append)
    assert not common.compatible_report({}, None)
    report = dict(run_id="one", selected_model="RF", test_results={}, configuration={}, data={})
    assert not common.compatible_report(report, {"run_id": "two"})
    assert common.compatible_report(report, {"run_id": "one"})
    assert len(errors) == len(recovery) == 1


def test_entrypoint_tracks_unhandled_render_failure(tmp_path, monkeypatch, caplog):
    import logging
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from heart_disease import config
    from heart_disease.ui import pages

    def broken_page(*args):
        raise RuntimeError("private input sentinel")

    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "missing")
    monkeypatch.setattr(pages, "overview", broken_page)
    with caplog.at_level(logging.ERROR, logger="heart_disease.dashboard"):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert app.exception
    events = [record for record in caplog.records if record.name == "heart_disease.dashboard"]
    assert len(events) == 1
    assert events[0].exception_type == "RuntimeError"
    assert events[0].exc_info is None
    assert "private input sentinel" not in events[0].getMessage()
