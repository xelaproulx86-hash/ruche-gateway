"""Greffe hivebase : le GITHUB_TOKEN ne doit JAMAIS fuiter — ni dans argv
(visible via ps), ni dans l'erreur levée (GRAFT_STATUS est affiché
publiquement dans /health et l'UI)."""
import subprocess

import pytest

SECRET = "ghp_tres_secret_12345"


def test_clone_sans_token(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "GITHUB_TOKEN", "")
    with pytest.raises(RuntimeError, match="pas de GITHUB_TOKEN"):
        app_mod._clone_repo("/tmp/hive-test")


def test_echec_de_clone_ne_fuite_pas_le_token(app_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "GITHUB_TOKEN", SECRET)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env", {})
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: auth")

    monkeypatch.setattr(app_mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        app_mod._clone_repo(str(tmp_path / "hive"))
    assert SECRET not in str(exc.value)          # l'erreur exposée est assainie
    assert all(SECRET not in part for part in seen["cmd"])  # rien dans argv
    assert SECRET in seen["env"].get("GIT_CONFIG_VALUE_0", "")  # token via env git


def test_clone_reussi_retourne_dst(app_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "GITHUB_TOKEN", SECRET)
    monkeypatch.setattr(
        app_mod.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    dst = str(tmp_path / "hive")
    assert app_mod._clone_repo(dst) == dst
