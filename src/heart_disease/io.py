"""Small, strict JSON and checksum utilities shared by scripts and the app."""

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_REPLACE_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Replace one document after serialization and a complete staged write.

    Readers see either the previous document or the complete replacement.
    Concurrent writers use separate temporary files; the last replace wins.
    This is a per-file guarantee, not a transaction across multiple artifacts.
    """
    payload = json.dumps(value, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows can reject two simultaneous replaces of the same destination.
        # Serialize this final step within the process; staging remains independent.
        with _REPLACE_LOCK:
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
