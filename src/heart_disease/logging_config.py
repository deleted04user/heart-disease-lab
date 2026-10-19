"""Idempotent JSON logging for CLI jobs and Streamlit reruns."""

import json
import logging
import os
from datetime import datetime, timezone

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
        return json.dumps(payload, ensure_ascii=False)


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
