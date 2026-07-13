"""Mode conseil : fan-out parallèle sur N cerveaux gratuits + arbitre."""

MSGS = [{"role": "user", "content": "capitale de la France ?"}]


def _arm(monkeypatch, app_mod, names):
    for prov in app_mod.FREE_CHAIN:
        if prov.name in names:
            monkeypatch.setenv(prov.key_env, "k")


def test_conseil_synthetise_en_parallele(monkeypatch, app_mod):
    _arm(monkeypatch, app_mod, {"sambanova", "cerebras", "gemini", "mistral"})
    advisor_calls = []

    def fake(base_url, api_key, model, messages, max_tokens=800):
        if "arbitre" in messages[0]["content"]:
            return "SYNTHÈSE: Paris"
        advisor_calls.append(base_url)
        return f"avis de {model}"

    monkeypatch.setattr(app_mod, "_call_openai_compat", fake)
    text, prov = app_mod.llm(MSGS, tier="conseil")
    assert text == "SYNTHÈSE: Paris"
    assert prov.startswith("conseil(") and "→" in prov
    assert len(advisor_calls) == app_mod.CONSEIL_ADVISORS  # 3 consultés en parallèle


def test_arbitre_choisi_hors_conseillers(monkeypatch, app_mod):
    _arm(monkeypatch, app_mod, {"sambanova", "cerebras", "gemini", "mistral"})

    def fake(base_url, api_key, model, messages, max_tokens=800):
        return "OK" if "arbitre" in messages[0]["content"] else "avis"

    monkeypatch.setattr(app_mod, "_call_openai_compat", fake)
    _, prov = app_mod.llm(MSGS, tier="conseil")
    # 3 premiers = conseillers ; l'arbitre doit être le 4e (mistral)
    assert prov.endswith("→mistral")


def test_repli_si_moins_de_deux_conseillers(monkeypatch, app_mod):
    _arm(monkeypatch, app_mod, {"sambanova"})
    monkeypatch.setattr(app_mod, "_call_openai_compat",
                        lambda *a, **k: "réponse unique")
    text, prov = app_mod.llm(MSGS, tier="conseil")
    assert text == "réponse unique"
    assert not prov.startswith("conseil")  # repli sur la cascade auto


def test_conseil_repli_si_tous_echouent(monkeypatch, app_mod):
    _arm(monkeypatch, app_mod, {"sambanova", "cerebras", "gemini"})

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(app_mod, "_call_openai_compat", boom)
    # tous les conseillers échouent → repli auto → free épuisé, GPU absent → erreur
    import pytest
    with pytest.raises(RuntimeError):
        app_mod.llm(MSGS, tier="conseil")


def test_conseil_via_ruche_chat(monkeypatch, app_mod):
    _arm(monkeypatch, app_mod, {"sambanova", "cerebras", "gemini"})

    def fake(base_url, api_key, model, messages, max_tokens=800):
        return "SYNTH" if "arbitre" in messages[0]["content"] else "avis"

    monkeypatch.setattr(app_mod, "_call_openai_compat", fake)
    out = app_mod.ruche_chat("salut", tier="conseil", token="test-token")
    assert "SYNTH" in out and "conseil(" in out
