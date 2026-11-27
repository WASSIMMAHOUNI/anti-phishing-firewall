"""
conftest.py — shared fixtures for PHISHGUARD tests.
"""
import sys
import json
from pathlib import Path
import pytest

# Make phishguard_app/ importable
_APP_DIR = Path(__file__).parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Repo root (for agent1_binary_outputs)
_REPO_ROOT = _APP_DIR.parent
AGENT1_DIR = _REPO_ROOT / "agent1_binary_outputs"
REFERENCE_EXAMPLES_PATH = AGENT1_DIR / "reference_examples.json"


@pytest.fixture(scope="session")
def reference_examples():
    if not REFERENCE_EXAMPLES_PATH.exists():
        pytest.skip(f"reference_examples.json not found at {REFERENCE_EXAMPLES_PATH}")
    with open(REFERENCE_EXAMPLES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def agent1_adapter():
    from agents.agent1_adapter import Agent1Adapter
    return Agent1Adapter(AGENT1_DIR)


@pytest.fixture(scope="session")
def agent2_adapter(tmp_path_factory):
    from agents.agent2_adapter import Agent2Adapter
    # Use a path that deliberately doesn't exist with auto_download=False to test unavailable path
    missing_dir = tmp_path_factory.mktemp("missing_agent2") / "nonexistent_model"
    return Agent2Adapter(
        model_dir=missing_dir,
        script_dir=_REPO_ROOT,
        auto_download=False,
    )


@pytest.fixture(scope="session")
def agent3_adapter(tmp_path_factory):
    from agents.agent3_adapter import Agent3Adapter
    missing_dir = tmp_path_factory.mktemp("missing_agent3") / "nonexistent_a3"
    return Agent3Adapter(missing_dir)


@pytest.fixture(scope="session")
def live_agent3_adapter():
    from agents.agent3_adapter import Agent3Adapter
    return Agent3Adapter(_REPO_ROOT / "agent3_outputs")
