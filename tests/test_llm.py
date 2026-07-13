"""Cascade économique : ordre, cooldown, épuisement, tiers."""
import pytest


MSGS = [{"role": "user", "content": "salut"}]


def _arm_providers(monkeypatch, app_mod, results):
    """Donne une clé aux providers nommés et scripte _call_openai_compat.
    results: {provider_name: str (réponse) | Exception (échec)}."""
    by_base = {}
    for prov in app_mod.FREE_CHAIN:
        if prov.name in results:
            monkeypatch.setenv(prov.key_env, "fake-key")
            by_base[prov.base_url] = results[prov.name]

    def fake_call(base_url, api_key, model, messages, max_tokens=800):
        out = by_base[base_url]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(app_mod, "_call_openai_compat", fake_call)


def test_cascade_prend_le_premier_provider_sain(monkeypatch, app_mod):
    _arm_providers(monkeypatch, app_mod, {"sambanova": "réponse A", "cerebras": "réponse B"})
    text, prov = app_mod.llm(MSGS, tier="free")
    assert (text, prov) == ("réponse A", "sambanova")


def test_cascade_bascule_et_met_en_cooldown(monkeypatch, app_mod):
    _arm_providers(monkeypatch, app_mod,
                   {"sambanova": RuntimeError("429"), "cerebras": "réponse B"})
    text, prov = app_mod.llm(MSGS, tier="free")
    assert (text, prov) == ("réponse B", "cerebras")
    assert app_mod._COOLDOWN["sambanova"] > 0


def test_provider_en_cooldown_est_saute(monkeypatch, app_mod):
    import time
    _arm_providers(monkeypatch, app_mod, {"sambanova": "réponse A", "cerebras": "réponse B"})
    app_mod._COOLDOWN["sambanova"] = time.time() + 60
    _, prov = app_mod.llm(MSGS, tier="free")
    assert prov == "cerebras"


def test_tier_free_epuise_leve_une_erreur(monkeypatch, app_mod):
    _arm_providers(monkeypatch, app_mod, {"sambanova": RuntimeError("boom")})
    with pytest.raises(RuntimeError, match="épuisée"):
        app_mod.llm(MSGS, tier="free")


def test_tier_gpu_non_configure(monkeypatch, app_mod):
    monkeypatch.setattr(app_mod, "RUNPOD_BASE_URL", "")
    monkeypatch.setattr(app_mod, "RUNPOD_API_KEY", "")
    with pytest.raises(RuntimeError, match="GPU non configuré"):
        app_mod.llm(MSGS, tier="gpu")


def test_tier_gpu_configure_appelle_runpod(monkeypatch, app_mod):
    monkeypatch.setattr(app_mod, "RUNPOD_BASE_URL", "https://runpod.example/v1")
    monkeypatch.setattr(app_mod, "RUNPOD_API_KEY", "rp-key")
    monkeypatch.setattr(app_mod, "_call_openai_compat",
                        lambda *a, **k: "réponse GPU")
    text, prov = app_mod.llm(MSGS, tier="gpu")
    assert (text, prov) == ("réponse GPU", "runpod-gpu")


def test_tier_inconnu(app_mod):
    with pytest.raises(RuntimeError, match="tier inconnu"):
        app_mod.llm(MSGS, tier="platine")


def test_sans_cle_aucun_provider_disponible(app_mod):
    assert app_mod._free_available() == []
