"""Reserve an official cohort as opaque bytes and prevent accidental evaluation."""

from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT
from .io import read_json, sha256, write_json

RESERVED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
)
RESERVED_PATH = ROOT / "data/reserved/processed.va.data"
RESERVATION = ROOT / "reports/reserved_cohort.json"


def reserve_cohort(
    protocol: Path = ROOT / "docs/next_experiment_protocol.md",
    path: Path = RESERVED_PATH,
    manifest: Path = RESERVATION,
) -> dict:
    # Only the acquisition helper is imported; never parse the reserved contents.
    from .data import _fetch

    if not protocol.is_file():
        raise ValueError("Declare a protocol before acquiring the reserved cohort.")
    protocol_hash = sha256(protocol)
    if manifest.exists():
        existing = read_json(manifest)
        if sha256(path) != existing["sha256"] or protocol_hash != existing["protocol_sha256"]:
            raise ValueError("Existing reservation or protocol differs; no bytes were overwritten.")
        return existing
    if path.exists():
        raise ValueError("Untracked reserved destination already exists; no data was overwritten.")
    content = _fetch(RESERVED_URL)
    if not content:
        raise ValueError("The reserved source returned no bytes.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
    result = {
        "status": "reserved_not_inspected",
        "source_url": RESERVED_URL,
        "path": "data/reserved/processed.va.data",
        "bytes": len(content),
        "sha256": sha256(path),
        "protocol": "docs/next_experiment_protocol.md",
        "protocol_sha256": protocol_hash,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "labels_inspected": False,
        "schema_inspected": False,
        "model_evaluated": False,
        "purpose": "Future fixed-protocol confirmation only; not included in training, diagnostics, release bundles or public Git history",
    }
    write_json(manifest, result)
    return result


def ensure_not_reserved(path: Path, manifest: Path = RESERVATION):
    path = Path(path).resolve()
    if path.is_relative_to((ROOT / "data/reserved").resolve()):
        raise ValueError("This cohort is reserved and not released for inspection/evaluation.")
    if manifest.is_file() and path.is_file():
        reservation = read_json(manifest)
        if reservation.get("status") == "reserved_not_inspected" and sha256(
            path
        ) == reservation.get("sha256"):
            raise ValueError(
                "These bytes match the untouched reserved cohort; ordinary loading/evaluation is refused."
            )
