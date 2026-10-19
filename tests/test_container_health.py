"""The process-health helper retries local failures and rejects external targets."""

import importlib.util
import io
from pathlib import Path
from urllib.error import URLError

import pytest

spec = importlib.util.spec_from_file_location(
    "container_health", Path(__file__).resolve().parents[1] / "scripts/verify_container.py"
)
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


@pytest.mark.parametrize("url", ["https://localhost:8502", "http://example.invalid:8502"])
def test_health_probe_rejects_external_or_tls_targets(url):
    with pytest.raises(ValueError, match="local HTTP"):
        health.wait_for_health(url)


def test_health_probe_retries_startup_failure(monkeypatch):
    calls = []

    def response(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise URLError("starting")
        result = io.BytesIO(b"ok")
        result.status = 200
        return result

    monkeypatch.setattr(health, "urlopen", response)
    monkeypatch.setattr(health.time, "sleep", lambda seconds: None)
    health.wait_for_health("http://127.0.0.1:8502", timeout=1)
    assert calls == ["http://127.0.0.1:8502/_stcore/health"] * 2


def test_health_probe_has_a_finite_deadline(monkeypatch):
    monkeypatch.setattr(
        health, "urlopen", lambda *args, **kwargs: pytest.fail("No request expected")
    )
    with pytest.raises(RuntimeError, match="deadline"):
        health.wait_for_health("http://127.0.0.1:8502", timeout=0)
