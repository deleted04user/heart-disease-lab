"""Idempotent JSON logging for CLI jobs and Streamlit reruns."""

import json
import logging
import os
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

_LEVELS = {
    name: getattr(logging, name) for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
}


class JsonFormatter(logging.Formatter):
    """Serialize operational messages; callers must not log patient inputs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for field in ("event_id", "exception_type", "frames"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False)


@contextmanager
def track_dashboard_errors():
    """Record failure locations without exception text, source lines or local values."""
    try:
        yield
    except Exception as exc:
        logging.getLogger("heart_disease.dashboard").error(
            "Unhandled dashboard failure",
            extra={
                "event_id": uuid4().hex,
                "exception_type": type(exc).__name__,
                "frames": [
                    {"file": os.path.basename(frame.filename), "line": frame.lineno}
                    for frame in traceback.extract_tb(exc.__traceback__)
                ],
            },
        )
        raise


def configure_logging(level: str | None = None) -> None:
    """Apply HEART_LOG_LEVEL and install one handler without replacing host handlers."""
    name = (level or os.environ.get("HEART_LOG_LEVEL", "INFO")).upper()
    if name not in _LEVELS:
        raise ValueError(f"Invalid HEART_LOG_LEVEL {name!r}; choose {', '.join(_LEVELS)}.")
    root = logging.getLogger()
    if not any(handler.get_name() == "heart-disease-json" for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.set_name("heart-disease-json")
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(_LEVELS[name])
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
