"""Vault : persistance git optionnelle de la mémoire. Sans VAULT_REPO,
comportement /tmp inchangé; le token ne fuite jamais (statut public)."""
import subprocess

SECRET = "ghp_vault_secret_999"


def _ok(cmd, **kw):
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def test_desactive_sans_vault_repo(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "VAULT_REPO", "")
    assert "désactivé" in app_mod.vault_init()
    assert app_mod._VAULT_ACTIVE is False


def test_desactive_sans_github_token(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "VAULT_REPO", "x/vault")
    monkeypatch.setattr(app_mod, "GITHUB_TOKEN", "")
    assert "GITHUB_TOKEN absent" in app_mod.vault_init()


def test_init_reussi_deplace_mem_file(app_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "VAULT_REPO", "x/vault")
    monkeypatch.setattr(app_mod, "GITHUB_TOKEN", SECRET)
    monkeypatch.setattr(app_mod, "VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setattr(app_mod, "_VAULT_ACTIVE", False)
    seen = []

    def fake_run(cmd, **kw):
        seen.append((cmd, kw.get("env", {})))
        return _ok(cmd)

    monkeypatch.setattr(app_mod.subprocess, "run", fake_run)
    status = app_mod.vault_init()
    assert "vault actif" in status
    assert app_mod._VAULT_ACTIVE is True
    assert app_mod.MEM_FILE.endswith("memory.json")
    # le token passe par l'env git, jamais par argv
    for cmd, env in seen:
        assert all(SECRET not in str(part) for part in cmd)
    assert any(SECRET in env.get("GIT_CONFIG_VALUE_0", "") for _, env in seen)


def test_echec_de_clone_statut_assaini_et_repli_tmp(app_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "VAULT_REPO", "x/vault")
    monkeypatch.setattr(app_mod, "GITHUB_TOKEN", SECRET)
    monkeypatch.setattr(app_mod, "VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setattr(app_mod, "_VAULT_ACTIVE", False)
    old_mem = app_mod.MEM_FILE
    monkeypatch.setattr(
        app_mod.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 128, stdout="",
                                                      stderr=f"fatal: {SECRET}"))
    status = app_mod.vault_init()
    assert "échec" in status and SECRET not in status  # statut public assaini
    assert app_mod._VAULT_ACTIVE is False
    assert app_mod.MEM_FILE == old_mem  # repli: mémoire /tmp inchangée


def test_remember_pousse_vers_le_vault(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "_VAULT_ACTIVE", True)
    monkeypatch.setattr(app_mod, "GITHUB_TOKEN", SECRET)
    pushed = []
    monkeypatch.setattr(app_mod.subprocess, "run",
                        lambda cmd, **kw: (pushed.append(cmd), _ok(cmd))[1])
    app_mod._mem_set("clé", "valeur")
    ops = [cmd[3] for cmd in pushed]  # ["git", "-C", dir, <op>, ...]
    assert ops == ["add", "commit", "push"]
    assert app_mod._mem() == {"clé": "valeur"}  # la copie locale est écrite


def test_vault_inactif_aucun_appel_git(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "_VAULT_ACTIVE", False)

    def boom(cmd, **kw):
        raise AssertionError("git ne doit pas être appelé")

    monkeypatch.setattr(app_mod.subprocess, "run", boom)
    app_mod._mem_set("a", "b")  # ne doit pas lever
    assert app_mod._mem() == {"a": "b"}


def test_echec_de_push_nest_pas_bloquant(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "_VAULT_ACTIVE", True)
    monkeypatch.setattr(app_mod, "GITHUB_TOKEN", SECRET)

    def fake_run(cmd, **kw):
        if cmd[3] == "push":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="réseau down")
        return _ok(cmd)

    monkeypatch.setattr(app_mod.subprocess, "run", fake_run)
    app_mod._mem_set("x", "y")  # ne doit pas lever
    assert app_mod._mem() == {"x": "y"}
