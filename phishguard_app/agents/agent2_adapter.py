"""
Agent 2 adapter — DistilBERT text classifier.

Supports the latest text_agent.py contract (commit 878c25b on feature/transformer-model):
  predict_email(subject: str, body: str, model_dir: Optional[str] = None) -> float
and falls back to TextAgent(model_dir=...).score_email(subject, body) if needed.

If local model weights are missing and auto_download=True, downloads the
archive directly to disk from the public GitHub Release v1.0-agent2-model,
verifies its SHA-256 checksum, and extracts it safely. Never fabricates scores.
"""
import hashlib
import math
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_STATUS_READY       = "ready"
_STATUS_UNAVAILABLE = "unavailable"

DEFAULT_MODEL_URL = (
    "https://github.com/WASSIMMAHOUNI/anti-phishing-firewall/"
    "releases/download/v1.0-agent2-model/distilbert_phishing_model.zip"
)
DEFAULT_MODEL_SHA256 = (
    "f0d715fa7f53a8bad16b5d37315b18f6001ade7dbe88ecd7b338c74fdc711a38"
)
_REQUIRED_MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


class Agent2Adapter:
    """
    Wraps Agent 2's predict_email() / TextAgent with graceful degradation
    and verified on-demand weight download for cloud deployments.
    """

    def __init__(
        self,
        model_dir: str | Path,
        script_dir: str | Path,
        auto_download: bool = True,
        download_url: str = DEFAULT_MODEL_URL,
        expected_sha256: str = DEFAULT_MODEL_SHA256,
    ):
        self._model_dir       = Path(model_dir).resolve()
        self._script_dir      = Path(script_dir).resolve()
        self._auto_download   = bool(auto_download)
        self._download_url    = download_url
        self._expected_sha256 = expected_sha256.lower()
        self._predict_fn      = None
        self._agent           = None
        self._status          = _STATUS_UNAVAILABLE
        self._error: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # Initialisation & verified download
    # ------------------------------------------------------------------

    @staticmethod
    def _has_valid_local_model(directory: Path) -> bool:
        return directory.is_dir() and all(
            (directory / fname).is_file() and (directory / fname).stat().st_size > 0 for fname in _REQUIRED_MODEL_FILES
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest().lower()

    @staticmethod
    def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
        dest_resolved = dest_dir.resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                target = (dest_resolved / member.filename).resolve()
                if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
                    raise ValueError(f"Unsafe zip member path detected: {member.filename!r}")
            zf.extractall(dest_resolved)

    def _resolve_writable_model_dir(self) -> Path:
        for candidate in (
            self._model_dir,
            Path(tempfile.gettempdir()) / "phishguard_models" / "distilbert_phishing_model",
        ):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write_test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                return candidate
            except OSError:
                continue
        raise OSError("No writable directory available to store Agent 2 model weights.")

    def _download_and_extract(self) -> Path:
        target_dir = self._resolve_writable_model_dir()
        if self._has_valid_local_model(target_dir):
            return target_dir

        part_path = target_dir.parent / f"{target_dir.name}.zip.part"
        zip_path  = target_dir.parent / f"{target_dir.name}.zip"
        stage_dir = target_dir.parent / f"{target_dir.name}_staging"

        for p in (part_path, zip_path):
            if p.exists():
                p.unlink()
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

        try:
            logger.info("Downloading Agent 2 weights from %s", self._download_url)
            req = urllib.request.Request(
                self._download_url,
                headers={"User-Agent": "PHISHGUARD/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp, part_path.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            actual_sha = self._sha256_file(part_path)
            if actual_sha != self._expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for downloaded Agent 2 archive: "
                    f"expected {self._expected_sha256}, got {actual_sha}"
                )

            part_path.replace(zip_path)
            stage_dir.mkdir(parents=True, exist_ok=True)
            self._safe_extract_zip(zip_path, stage_dir)

            if not self._has_valid_local_model(stage_dir):
                raise FileNotFoundError(
                    "Extracted archive does not contain required model files "
                    f"({', '.join(_REQUIRED_MODEL_FILES)})."
                )

            for item in stage_dir.iterdir():
                dest = target_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))

            return target_dir
        finally:
            for p in (part_path, zip_path):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)

    def _load(self):
        # 1. Check torch + transformers importable
        try:
            import torch               # noqa: F401
            import transformers        # noqa: F401
        except ImportError as exc:
            self._error = (
                f"Missing dependency — {exc}. "
                "Install torch and transformers to enable Agent 2."
            )
            return

        # 2. Reuse existing local model or download & verify if missing
        active_model_dir = self._model_dir
        if not self._has_valid_local_model(active_model_dir):
            fallback_dir = (
                Path(tempfile.gettempdir()) / "phishguard_models" / "distilbert_phishing_model"
            )
            if self._has_valid_local_model(fallback_dir):
                active_model_dir = fallback_dir
            elif self._auto_download:
                try:
                    active_model_dir = self._download_and_extract()
                except Exception as exc:
                    self._error = f"Agent 2 model download/verification failed: {exc}"
                    logger.exception("Agent 2 weight download failed")
                    return
            else:
                self._error = (
                    f"Model weights not found at '{self._model_dir}'. "
                    "Download distilbert_phishing_model.zip (release v1.0-agent2-model) "
                    "and extract it there."
                )
                return

        self._model_dir = active_model_dir

        # 3. Check text_agent.py exists in script_dir
        script_path = self._script_dir / "text_agent.py"
        if not script_path.exists():
            self._error = (
                f"Inference module not found at '{script_path}'. "
                "Bring text_agent.py from branch feature/transformer-model."
            )
            return

        # 4. Import and verify predict_email (or backward-compatible TextAgent)
        try:
            sd = str(self._script_dir)
            if sd not in sys.path:
                sys.path.insert(0, sd)
            import text_agent  # noqa: PLC0415

            if hasattr(text_agent, "predict_email"):
                if hasattr(text_agent, "_load_artifacts"):
                    text_agent._load_artifacts(str(self._model_dir))
                self._predict_fn = text_agent.predict_email
                self._status = _STATUS_READY
            elif hasattr(text_agent, "TextAgent"):
                self._agent = text_agent.TextAgent(model_dir=str(self._model_dir))
                self._status = _STATUS_READY
            else:
                self._error = "text_agent.py exposes neither predict_email nor TextAgent."
        except Exception as exc:
            self._error = f"Agent 2 load error: {exc}"
            logger.exception("Agent 2 load failed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "name":   "Agent 2 — DistilBERT Text Classifier",
            "status": self._status,
            "error":  self._error,
        }

    def run(self, subject: str, body: str) -> Dict[str, Any]:
        """
        Score one email's text content.

        Returns
        -------
        dict
          available   bool
          status      'completed' | 'unavailable' | 'failed'
          text_score  float in [0,1], P(Not Safe) — present if completed
          error       str — present if not completed
        """
        if self._status != _STATUS_READY:
            return {
                "available": False,
                "status":    "unavailable",
                "error":     self._error,
            }

        try:
            if self._predict_fn is not None:
                score = self._predict_fn(
                    subject or "",
                    body or "",
                    model_dir=str(self._model_dir),
                )
            else:
                score = self._agent.score_email(subject or "", body or "")
            return {
                "available":  True,
                "status":     "completed",
                "text_score": float(score) if math.isfinite(float(score)) else (_ for _ in ()).throw(ValueError("Non-finite P(Not Safe)")),
            }
        except Exception as exc:
            logger.exception("Agent 2 inference error")
            return {
                "available": False,
                "status":    "failed",
                "error":     str(exc),
            }
