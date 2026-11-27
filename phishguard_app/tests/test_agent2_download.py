"""
Tests for Agent 2 verified model download, checksum validation,
corrupted/incomplete archive handling, local reuse, and 3-agent startup.
"""
import hashlib
import io
import shutil
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent2_adapter import (
    DEFAULT_MODEL_SHA256,
    DEFAULT_MODEL_URL,
    Agent2Adapter,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
_REAL_MODEL_DIR = _REPO_ROOT / "models" / "distilbert_phishing_model"


class _FakeResponse:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_model_zip_bytes() -> tuple[bytes, str]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", '{"model_type": "distilbert"}')
        zf.writestr("model.safetensors", "dummy-weights")
        zf.writestr("tokenizer.json", "{}")
        zf.writestr("tokenizer_config.json", "{}")
    raw = buf.getvalue()
    return raw, hashlib.sha256(raw).hexdigest()


def test_default_release_url_and_sha256_constants():
    assert "v1.0-agent2-model/distilbert_phishing_model.zip" in DEFAULT_MODEL_URL
    assert DEFAULT_MODEL_SHA256 == (
        "f0d715fa7f53a8bad16b5d37315b18f6001ade7dbe88ecd7b338c74fdc711a38"
    )


def test_missing_model_downloads_verifies_checksum_and_extracts(tmp_path):
    zip_bytes, sha = _make_model_zip_bytes()
    target_dir = tmp_path / "models" / "distilbert_phishing_model"

    with (
        patch("urllib.request.urlopen", return_value=_FakeResponse(zip_bytes)) as mock_open,
        patch("text_agent._load_artifacts", return_value=(None, None, None)),
    ):
        adapter = Agent2Adapter(
            model_dir=target_dir,
            script_dir=_REPO_ROOT,
            auto_download=True,
            expected_sha256=sha,
        )

    assert mock_open.call_count == 1
    assert adapter.status()["status"] == "ready"
    assert (target_dir / "config.json").is_file()
    assert (target_dir / "model.safetensors").is_file()
    assert not (tmp_path / "models" / "distilbert_phishing_model.zip").exists()
    assert not (tmp_path / "models" / "distilbert_phishing_model.zip.part").exists()


def test_corrupted_checksum_fails_and_reports_unavailable(tmp_path, monkeypatch):
    zip_bytes, _ = _make_model_zip_bytes()
    target_dir = tmp_path / "models" / "distilbert_phishing_model"
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "tmp"))

    with patch("urllib.request.urlopen", return_value=_FakeResponse(zip_bytes)):
        adapter = Agent2Adapter(
            model_dir=target_dir,
            script_dir=_REPO_ROOT,
            auto_download=True,
            expected_sha256="0" * 64,
        )

    s = adapter.status()
    assert s["status"] == "unavailable"
    assert "SHA-256 mismatch" in s["error"]
    assert not (target_dir / "model.safetensors").exists()
    assert adapter.run("Subj", "Body")["available"] is False


def test_incomplete_archive_fails_and_reports_unavailable(tmp_path, monkeypatch):
    incomplete = b"PK\x03\x04truncated-archive-bytes"
    sha = hashlib.sha256(incomplete).hexdigest()
    target_dir = tmp_path / "models" / "distilbert_phishing_model"
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "tmp"))

    with patch("urllib.request.urlopen", return_value=_FakeResponse(incomplete)):
        adapter = Agent2Adapter(
            model_dir=target_dir,
            script_dir=_REPO_ROOT,
            auto_download=True,
            expected_sha256=sha,
        )

    s = adapter.status()
    assert s["status"] == "unavailable"
    assert adapter.run("Subj", "Body")["available"] is False
    assert not (tmp_path / "models" / "distilbert_phishing_model.zip.part").exists()


def test_existing_model_reused_without_downloading(tmp_path):
    target_dir = tmp_path / "models" / "distilbert_phishing_model"
    target_dir.mkdir(parents=True)
    shutil.copy2(_REAL_MODEL_DIR / "config.json", target_dir / "config.json")
    shutil.copy2(_REAL_MODEL_DIR / "model.safetensors", target_dir / "model.safetensors")
    shutil.copy2(_REAL_MODEL_DIR / "tokenizer.json", target_dir / "tokenizer.json")
    shutil.copy2(_REAL_MODEL_DIR / "tokenizer_config.json", target_dir / "tokenizer_config.json")

    with (
        patch("urllib.request.urlopen") as mock_open,
        patch("text_agent._load_artifacts", return_value=(None, None, None)),
    ):
        adapter = Agent2Adapter(
            model_dir=target_dir,
            script_dir=_REPO_ROOT,
            auto_download=True,
        )

    assert mock_open.call_count == 0
    assert adapter.status()["status"] == "ready"


def test_application_startup_with_all_three_agents():
    from orchestrator import Orchestrator

    orch = Orchestrator()
    statuses = orch.get_all_status()
    assert statuses["agent1"]["status"] == "ready"
    assert statuses["agent2"]["status"] == "ready"
    assert statuses["agent3"]["status"] == "ready"
