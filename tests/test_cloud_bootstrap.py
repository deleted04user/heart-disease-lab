"""Checksum and cache boundaries with in-memory archives, never external calls."""

import io
import json
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

import heart_disease.bundle as bundle


def archive_bytes(files=None):
    files = files or {"app.py": b"isolated_fixture = True\n"}
    manifest = {"schema": 1, "files": {name: bundle.checksum(data) for name, data in files.items()}}
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("bundle_manifest.json", json.dumps(manifest))
    return stream.getvalue()


def cloud_config(tmp_path, payload):
    with ZipFile(io.BytesIO(payload)) as archive:
        manifest_digest = bundle.checksum(archive.read("bundle_manifest.json"))
    value = {
        "repository": "test-only/fixture",
        "version": "test",
        "asset": "fixture.zip",
        "sha256": bundle.checksum(payload),
        "manifest_sha256": manifest_digest,
        "url": "https://github.com/test-only/fixture/releases/download/vtest/fixture.zip",
    }
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value))
    return path, value


def test_verified_cache_reuses_download_and_detects_tampering(tmp_path, monkeypatch):
    payload = archive_bytes()
    config, _ = cloud_config(tmp_path, payload)
    requests = []

    def download(request, timeout):
        requests.append(request.full_url)
        assert timeout == 45
        return io.BytesIO(payload)

    monkeypatch.setattr(bundle, "urlopen", download)
    cache = tmp_path / "cache"
    destination = bundle.ensure_cloud_bundle(config, cache)
    assert bundle.ensure_cloud_bundle(config, cache) == destination
    assert len(requests) == 1
    (destination / "app.py").write_text("tampered")
    with pytest.raises(ValueError, match="damaged"):
        bundle.ensure_cloud_bundle(config, cache)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "http://untrusted.invalid/fixture"),
        ("sha256", "bad"),
        ("manifest_sha256", "bad"),
    ],
)
def test_invalid_cloud_configuration_never_opens_network(tmp_path, monkeypatch, field, value):
    config, values = cloud_config(tmp_path, archive_bytes())
    values[field] = value
    config.write_text(json.dumps(values))

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid configuration reached the network")

    monkeypatch.setattr(bundle, "urlopen", unexpected)
    with pytest.raises(ValueError):
        bundle.ensure_cloud_bundle(config, tmp_path / "cache")


def test_failed_download_cannot_publish_a_partial_cache(tmp_path, monkeypatch):
    config, _ = cloud_config(tmp_path, archive_bytes())

    def unavailable(*args, **kwargs):
        raise TimeoutError("offline fixture")

    monkeypatch.setattr(bundle, "urlopen", unavailable)
    with pytest.raises(TimeoutError):
        bundle.ensure_cloud_bundle(config, tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_symlink_and_unlisted_files_are_rejected(tmp_path):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        link = ZipInfo("link")
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "outside")
    payload = stream.getvalue()
    with pytest.raises(ValueError, match="Symbolic links"):
        bundle.extract_bundle(payload, tmp_path / "links", bundle.checksum(payload))
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("extra.py", "unexpected")
        archive.writestr("bundle_manifest.json", json.dumps({"schema": 1, "files": {}}))
    payload = stream.getvalue()
    with pytest.raises(ValueError, match="empty bundle manifest"):
        bundle.extract_bundle(payload, tmp_path / "extra", bundle.checksum(payload))


def test_expansion_limit_is_checked_before_extraction(tmp_path, monkeypatch):
    payload = archive_bytes({"app.py": b"x" * 10_000})
    monkeypatch.setattr(bundle, "MAX_BUNDLE_BYTES", len(payload) + 10)
    with pytest.raises(ValueError, match="expanded archive size"):
        bundle.extract_bundle(payload, tmp_path / "large", bundle.checksum(payload))


def test_empty_manifest_is_rejected(tmp_path):
    (tmp_path / "bundle_manifest.json").write_text(json.dumps({"schema": 1, "files": {}}))
    with pytest.raises(ValueError, match="empty bundle manifest"):
        bundle.verify_directory(tmp_path)


@pytest.mark.parametrize("name", ["src/unlisted.py", "src/__pycache__/unlisted.pyc"])
def test_cached_bundle_rejects_unlisted_source_and_bytecode(tmp_path, monkeypatch, name):
    payload = archive_bytes()
    config, _ = cloud_config(tmp_path, payload)
    monkeypatch.setattr(bundle, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    destination = bundle.ensure_cloud_bundle(config, tmp_path / "cache")
    unexpected = destination / name
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_bytes(b"unlisted fixture")
    with pytest.raises(ValueError, match="inventory"):
        bundle.ensure_cloud_bundle(config, tmp_path / "cache")


@pytest.mark.parametrize(
    "files",
    [
        {"App.py": "0" * 64, "app.py": "0" * 64},
        {"Src/a.py": "0" * 64, "src/b.py": "0" * 64},
        {"app.py": "invalid digest"},
        {"bundle_manifest.json": "0" * 64},
    ],
)
def test_invalid_manifest_inventory_is_rejected_before_extraction(tmp_path, files):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name in files:
            if name != "bundle_manifest.json":
                archive.writestr(name, "fixture")
        archive.writestr("bundle_manifest.json", json.dumps({"schema": 1, "files": files}))
    payload = stream.getvalue()
    with pytest.raises(ValueError, match="manifest|digest|collid"):
        bundle.extract_bundle(payload, tmp_path / "invalid", bundle.checksum(payload))
    assert not list((tmp_path / "invalid").rglob("*.py"))


@pytest.mark.skipif(sys.platform == "win32", reason="Creating symlinks requires Windows privileges")
def test_cached_bundle_rejects_internal_symlinks(tmp_path):
    payload = archive_bytes()
    destination = tmp_path / "bundle"
    bundle.extract_bundle(payload, destination, bundle.checksum(payload))
    (destination / "unlisted-link").symlink_to(destination / "app.py")
    with pytest.raises(ValueError, match="Symbolic links"):
        bundle.verify_directory(destination)


def test_cache_cannot_redefine_its_own_trusted_manifest(tmp_path, monkeypatch):
    payload = archive_bytes()
    config, _ = cloud_config(tmp_path, payload)
    monkeypatch.setattr(bundle, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    cache = tmp_path / "cache"
    destination = bundle.ensure_cloud_bundle(config, cache)
    replacement = b"replaced source fixture"
    (destination / "app.py").write_bytes(replacement)
    manifest = {"schema": 1, "files": {"app.py": bundle.checksum(replacement)}}
    (destination / "bundle_manifest.json").write_text(json.dumps(manifest))
    bundle.verify_directory(destination)
    with pytest.raises(ValueError, match="trusted configuration"):
        bundle.ensure_cloud_bundle(config, cache)


def test_wrong_pinned_manifest_cannot_publish_downloaded_cache(tmp_path, monkeypatch):
    payload = archive_bytes()
    config, value = cloud_config(tmp_path, payload)
    value["manifest_sha256"] = "0" * 64
    config.write_text(json.dumps(value))
    monkeypatch.setattr(bundle, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="trusted configuration"):
        bundle.ensure_cloud_bundle(config, cache)
    assert list(cache.iterdir()) == []
