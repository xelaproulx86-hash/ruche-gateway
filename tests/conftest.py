"""Fixtures partagées. L'import de app.py est fait ici, une seule fois,
avec un environnement neutralisé (pas de clés provider, pas de réseau)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Environnement déterministe AVANT l'import de app.py
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["GATEWAY_TOKEN"] = "test-token"
os.environ.pop("GITHUB_TOKEN", None)      # greffe hivebase dormante
os.environ.pop("RUNPOD_BASE_URL", None)
os.environ.pop("RUNPOD_API_KEY", None)
for _p in ("SAMBANOVA", "CEREBRAS", "GEMINI", "MISTRAL", "NVIDIA", "GROQ"):
    os.environ.pop(f"{_p}_API_KEY", None)

import app as ruche  # noqa: E402


@pytest.fixture()
def app_mod():
    return ruche


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    """Chaque test part avec un cooldown vide et une mémoire isolée."""
    ruche._COOLDOWN.clear()
    monkeypatch.setattr(ruche, "MEM_FILE", str(tmp_path / "mem.json"))
    yield
    ruche._COOLDOWN.clear()
