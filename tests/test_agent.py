"""Extraction JSON et boucle d'agent penser→agir→observer."""


# ── _extract_json ────────────────────────────────────────────────────
def test_json_pur(app_mod):
    assert app_mod._extract_json('{"action": "done", "args": {}}') == \
        {"action": "done", "args": {}}


def test_json_entoure_de_prose(app_mod):
    raw = 'Voici ma réponse :\n```json\n{"action": "done", "args": {"answer": "ok"}}\n```\nVoilà !'
    assert app_mod._extract_json(raw) == {"action": "done", "args": {"answer": "ok"}}


def test_plusieurs_objets_prend_le_premier(app_mod):
    raw = '{"action": "recall", "args": {}} {"action": "done", "args": {}}'
    assert app_mod._extract_json(raw)["action"] == "recall"


def test_accolades_dans_les_chaines(app_mod):
    raw = '{"action": "remember", "args": {"key": "a", "value": "x } y { z"}}'
    assert app_mod._extract_json(raw)["args"]["value"] == "x } y { z"


def test_sans_json_retourne_none(app_mod):
    assert app_mod._extract_json("désolé, je ne peux pas") is None
    assert app_mod._extract_json("") is None
    assert app_mod._extract_json("{cassé") is None


def test_json_non_objet_retourne_none(app_mod):
    assert app_mod._extract_json('["une", "liste"]') is None


# ── _run_agent ───────────────────────────────────────────────────────
def _script_llm(app_mod, monkeypatch, responses):
    """llm scripté : renvoie les réponses dans l'ordre."""
    it = iter(responses)
    monkeypatch.setattr(app_mod, "llm",
                        lambda msgs, tier="auto", max_tokens=800: (next(it), "fake"))


def test_agent_execute_puis_termine(app_mod, monkeypatch):
    _script_llm(app_mod, monkeypatch, [
        '{"action": "remember", "args": {"key": "n", "value": "42"}}',
        '{"action": "done", "args": {"answer": "c\'est noté"}}',
    ])
    out = app_mod._run_agent("retiens 42")
    assert "c'est noté" in out
    assert "remember" in out
    assert app_mod._mem() == {"n": "42"}


def test_agent_reponse_non_json_rendue_brute(app_mod, monkeypatch):
    _script_llm(app_mod, monkeypatch, ["Je réponds en prose, sans JSON."])
    assert app_mod._run_agent("question") == "Je réponds en prose, sans JSON."


def test_agent_limite_de_5_actions(app_mod, monkeypatch):
    _script_llm(app_mod, monkeypatch,
                ['{"action": "recall", "args": {"key": "x"}}'] * 5)
    assert "Limite de 5 actions" in app_mod._run_agent("boucle")


def test_agent_tache_vide(app_mod):
    assert app_mod._run_agent("") == "Tâche vide."
    assert app_mod._run_agent("   ") == "Tâche vide."


def test_agent_tache_trop_longue(app_mod):
    out = app_mod._run_agent("x" * (app_mod.MAX_TASK_CHARS + 1))
    assert "trop longue" in out


def test_agent_erreur_llm_capturee(app_mod, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("chaîne gratuite épuisée")
    monkeypatch.setattr(app_mod, "llm", boom)
    assert "Erreur gateway" in app_mod._run_agent("tâche")


def test_ruche_chat_ok(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "llm",
                        lambda msgs, tier="free", max_tokens=800: ("bonjour!", "cerebras"))
    out = app_mod.ruche_chat("salut", token="test-token")
    assert "bonjour!" in out and "cerebras" in out


def test_ruche_chat_prompt_trop_long(app_mod):
    out = app_mod.ruche_chat("x" * (app_mod.MAX_TASK_CHARS + 1), token="test-token")
    assert "trop long" in out


def test_gateway_info_est_du_json(app_mod):
    import json
    info = json.loads(app_mod.gateway_info())
    assert "tiers" in info and "graft" in info
