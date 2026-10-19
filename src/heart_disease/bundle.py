"""Portable release archives and a checksum-pinned, standard-library bootstrap.

No project configuration or ML modules are imported until build time. The cloud
entrypoint can therefore locate its bundle before configuring model/report paths.
"""
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

MAX_BUNDLE_BYTES = 100_000_000


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute() or any(part in {".", ".."} for part in name.split("/")):
        raise ValueError(f"Unsafe bundle member: {name}")
    return path


def sanitize(value, workspace: Path):
    """Remove machine paths from public aggregate reports, preserving URLs."""
    if isinstance(value, dict):
        return {key: sanitize(item, workspace) for key, item in value.items() if key not in {"_artifact_dir", "immutable_run_dir", "reports_sha256"}}
    if isinstance(value, list):
        return [sanitize(item, workspace) for item in value]
    if isinstance(value, str):
        text = value.replace(str(workspace), "[training workspace]").replace(workspace.as_posix(), "[training workspace]")
        if re.match(r"^[A-Za-z]:[\\/]|^/(?:home|Users|tmp|mnt|workspace)/", text):
            return "[training workspace]/" + re.split(r"[\\/]", text)[-1]
        return text
    return value


def build_bundle(workspace: Path, model_path: Path, report_dir: Path, output: Path, *, version: str = "1.1.0", examples: int = 5) -> dict:
    from .config import FEATURES
    from .data import load_dataset
    from .prediction import load_artifact

    workspace, model_path, report_dir = Path(workspace).resolve(), Path(model_path).resolve(), Path(report_dir).resolve()
    pipeline, metadata = load_artifact(model_path)
    source = metadata["data"]
    if not (source["source"].get("url") or "").startswith("https://archive.ics.uci.edu/"):
        raise ValueError("Public releases require verified official UCI source provenance.")
    dataset = load_dataset(Path(source["path"]), schema=source["schema"], target_semantics=source["target_semantics"], site=source.get("site") or "Cleveland")
    if dataset.metadata["sha256"] != source["sha256"]:
        raise ValueError("Training source changed; refusing to publish example records.")
    complete = dataset.frame[FEATURES].dropna().head(examples)
    if examples < 1 or complete.empty:
        raise ValueError("At least one complete public demonstration record is required.")
    records = [{"source_row": int(index) + 1, "features": {col: float(row[col]) if col == "oldpeak" else int(row[col]) for col in FEATURES}} for index, row in complete.iterrows()]
    example = {"schema": 1, "source_sha256": source["sha256"], "source_url": source["source"]["url"], "license": "CC BY 4.0", "records": records}
    files = {}
    for name in ["app.py", "LICENSE", "NOTICE.md"]:
        files[name] = (workspace / name).read_bytes()
    for path in sorted((workspace / "src/heart_disease").glob("*.py")):
        files[path.relative_to(workspace).as_posix()] = path.read_bytes()
    for path in sorted(report_dir.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.name.endswith("_verification.json") or path.name in {"active_run.json", "verification_report.md", "publication_report.json"}:
            continue
        relative = "reports/" + path.relative_to(report_dir).as_posix()
        if path.suffix == ".json":
            files[relative] = json_bytes(sanitize(json.loads(path.read_text(encoding="utf-8")), workspace))
        elif path.suffix in {".md", ".csv", ".txt"}:
            files[relative] = sanitize(path.read_text(encoding="utf-8"), workspace).encode("utf-8")
        elif path.suffix == ".png":
            files[relative] = path.read_bytes()
    files["models/heart_model.joblib"] = model_path.read_bytes()
    files["models/public_examples.json"] = json_bytes(example)
    portable = sanitize(metadata, workspace)
    portable["data"]["path"] = "public-source/processed.cleveland.data (not bundled)"
    portable["report_dir"] = "reports"
    portable["public_demo"] = {"path": "public_examples.json", "sha256": checksum(files["models/public_examples.json"]), "count": len(records)}
    portable["release_version"] = version
    files["models/heart_model.metadata.json"] = json_bytes(portable)
    versions = metadata["dependency_versions"]
    required = ["numpy", "pandas", "scikit-learn", "streamlit", "joblib"]
    if metadata["selected_model"] == "XGBoost":
        required.append("xgboost")
    files["requirements.txt"] = ("\n".join(f"{name}=={versions[name]}" for name in required) + "\n").encode()
    files[".streamlit/config.toml"] = (workspace / ".streamlit/config.toml").read_bytes()
    files["README.md"] = f"""# Heart Disease Lab — portable educational release {version}

This archive contains a trusted fitted pipeline, sanitized aggregate reports,
five or fewer anonymous public UCI examples, and the application source.
The complete downloaded datasets and user input history are not included.

Create a Python 3.13 environment, install `requirements.txt`, then run:

    python -m streamlit run app.py --server.address 127.0.0.1

No retraining, Kaggle credentials, original workstation path or raw source file
is needed for the example loader. Use `HEART_PUBLIC_DEMO=1` to disable manual
measurement entry and restrict the form to bundled public examples.

Read models/heart_model.metadata.json and reports/model_card.md for provenance,
actual metrics and limitations. Load only trusted joblib artifacts: Python
deserialization can execute code. The bundle manifest detects damaged files;
the committed cloud release SHA pins the trusted published archive.

Code: MIT. Public examples: UCI Heart Disease, Janosi et al. (1989), CC BY 4.0,
https://doi.org/10.24432/C52P4X. See NOTICE.md. Not for medical decisions.
""".encode("utf-8")
    manifest = {"schema": 1, "version": version, "run_id": metadata["run_id"], "selected_model": metadata["selected_model"], "source_sha256": source["sha256"], "example_count": len(records), "files": {name: checksum(content) for name, content in sorted(files.items())}}
    files["bundle_manifest.json"] = json_bytes(manifest)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            safe_member(name)
            info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return {"version": version, "asset": output.name, "sha256": checksum(output.read_bytes()), "bytes": output.stat().st_size, "run_id": metadata["run_id"], "example_count": len(records), "runtime_requirements": files["requirements.txt"].decode(), "model_probability_smoke": float(pipeline.predict_proba(complete.iloc[[0]])[0, 1])}


def verify_directory(root: Path) -> dict:
    root = Path(root).resolve()
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty bundle manifest.")
    for name, expected in manifest["files"].items():
        safe_member(name)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or checksum(path.read_bytes()) != expected:
            raise ValueError(f"Missing or damaged bundle file: {name}")
    return manifest


def extract_bundle(payload: bytes, destination: Path, expected_sha256: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or checksum(payload) != expected_sha256:
        raise ValueError("Release archive SHA-256 differs from its trusted configuration.")
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError("Release archive exceeds size limit.")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        names = [info.filename for info in entries]
        if len(names) != len(set(names)) or sum(info.file_size for info in entries) > MAX_BUNDLE_BYTES:
            raise ValueError("Duplicate members or excessive expanded archive size.")
        for info in entries:
            safe_member(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symbolic links are not accepted in release archives.")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if set(names) != set(manifest.get("files", {})) | {"bundle_manifest.json"}:
            raise ValueError("Archive contents differ from manifest.")
        for info in entries:
            target = destination / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(archive.read(info.filename))
    return verify_directory(destination)


def ensure_cloud_bundle(config_path: Path, cache: Path) -> Path:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected_url = f"https://github.com/{config['repository']}/releases/download/v{config['version']}/{config['asset']}"
    if config.get("url") != expected_url or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config["repository"]):
        raise ValueError("Cloud configuration must pin an explicit GitHub repository release.")
    digest = config["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("A valid release checksum is required.")
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / digest
    if destination.exists():
        verify_directory(destination)
        return destination
    request = Request(expected_url, headers={"User-Agent": "HeartDiseaseLab/1.1 educational release bootstrap"})
    with urlopen(request, timeout=45) as response:
        payload = response.read(MAX_BUNDLE_BYTES + 1)
    with tempfile.TemporaryDirectory(prefix="release-", dir=cache) as staging:
        temporary = Path(staging) / "bundle"
        extract_bundle(payload, temporary, digest)
        try:
            temporary.rename(destination)
        except FileExistsError:
            verify_directory(destination)
    return destination
