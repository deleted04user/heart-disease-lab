import json
import logging

import pytest

from heart_disease.logging_config import configure_logging


@pytest.fixture
def isolated_root(monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)
    return root


def test_reruns_keep_one_handler_and_emit_json(isolated_root, capsys):
    host_handlers = list(isolated_root.handlers)
    configure_logging("info")
    configure_logging("INFO")
    assert len(isolated_root.handlers) == len(host_handlers) + 1
    assert all(handler in isolated_root.handlers for handler in host_handlers)
    logging.getLogger("heart_disease.test").info("Completed %d records", 3)
    record = json.loads(capsys.readouterr().err)
    assert record["level"] == "INFO"
    assert record["message"] == "Completed 3 records"
    assert record["logger"] == "heart_disease.test"
    assert record["timestamp"].endswith("+00:00")


def test_environment_level_filters_messages(isolated_root, monkeypatch, capsys):
    monkeypatch.setenv("HEART_LOG_LEVEL", "error")
    configure_logging()
    logger = logging.getLogger("heart_disease.test")
    logger.warning("filtered")
    logger.error("retained")
    assert json.loads(capsys.readouterr().err)["message"] == "retained"


def test_invalid_level_does_not_change_host_logging(isolated_root):
    previous = list(isolated_root.handlers)
    with pytest.raises(ValueError, match="Invalid HEART_LOG_LEVEL"):
        configure_logging("invalid")
    assert isolated_root.handlers == previous


def test_exception_context_is_json_encoded(isolated_root, capsys):
    configure_logging()
    try:
        raise RuntimeError("operational failure")
    except RuntimeError:
        logging.getLogger("heart_disease.test").exception("Job failed")
    assert "RuntimeError: operational failure" in json.loads(capsys.readouterr().err)["exception"]
