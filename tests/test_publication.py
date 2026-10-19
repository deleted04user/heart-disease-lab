"""Use a temporary Git index to verify publication exclusions and safe templates."""

import importlib
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("audit_publication")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    return module, tmp_path


def stage(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(["git", "add", "-f", name], cwd=root, check=True, capture_output=True)


def test_documented_environment_template_is_allowed(index):
    module, root = index
    stage(root, ".env.example", "HEART_LOG_LEVEL=INFO\n")
    assert module.audit()["status"] == "passed"


@pytest.mark.parametrize("name", [".env", ".env.production", "data/reserved/fixture.data"])
def test_private_environment_and_reserved_data_are_rejected(index, name):
    module, root = index
    stage(root, name, "isolated fixture")
    with pytest.raises(ValueError, match="Publication review failed"):
        module.audit()


def test_environment_template_still_undergoes_credential_scanning(index):
    module, root = index
    stage(root, ".env.example", "ACCESS=" + "gh" + "p_" + "x" * 36)
    with pytest.raises(ValueError, match="Possible credential"):
        module.audit()
