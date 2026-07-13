"""run_tool : actions intégrées, greffes hivebase, erreurs."""


def test_action_inconnue(app_mod):
    assert "action inconnue" in app_mod.run_tool("teleporter", {})


def test_remember_recall(app_mod):
    assert app_mod.run_tool("remember", {"key": "ville", "value": "Québec"}) == "mémorisé"
    assert app_mod.run_tool("recall", {"key": "ville"}) == "Québec"
    assert app_mod.run_tool("recall", {"key": "inconnue"}) == "(vide)"


def test_web_get_url_bloquee(app_mod):
    assert "bloquée" in app_mod.run_tool("web_get", {"url": "http://127.0.0.1/"})
    assert "invalide" in app_mod.run_tool("web_get", {"url": "file:///etc/passwd"})


def test_n8n_non_configure(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "N8N_WEBHOOK", "")
    assert "non configuré" in app_mod.run_tool("n8n", {"payload": {}})


def test_outil_greffe_est_appele(app_mod, monkeypatch):
    monkeypatch.setitem(app_mod.HIVE_TOOLS, "echo", lambda texte: f"echo:{texte}")
    assert app_mod.run_tool("echo", {"texte": "abc"}) == "echo:abc"


def test_outil_greffe_mauvais_args(app_mod, monkeypatch):
    monkeypatch.setitem(app_mod.HIVE_TOOLS, "echo", lambda texte: texte)
    out = app_mod.run_tool("echo", {"mauvais_param": "x"})
    assert "erreur args" in out


def test_outil_greffe_resultat_tronque(app_mod, monkeypatch):
    monkeypatch.setitem(app_mod.HIVE_TOOLS, "gros", lambda: "x" * 10000)
    assert len(app_mod.run_tool("gros", {})) <= 3000


def test_exception_outil_capturee(app_mod, monkeypatch):
    def boom():
        raise ValueError("cassé")
    monkeypatch.setitem(app_mod.HIVE_TOOLS, "boom", boom)
    assert "erreur outil" in app_mod.run_tool("boom", {})
