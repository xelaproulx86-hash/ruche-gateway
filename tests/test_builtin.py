"""Outils embarqués (vendored) : toujours présents sans token, et la
greffe s'ajoute par-dessus sans les écraser."""


def test_builtins_presents_sans_token(app_mod):
    for name in ("heure", "calculer", "meteo"):
        assert name in app_mod.HIVE_TOOLS
    assert set(app_mod.BUILTIN_TOOL_NAMES) == {"calculer", "heure", "meteo"}


def test_builtin_calculer(app_mod):
    assert app_mod.run_tool("calculer", {"expression": "6*7"}) == "42"
    assert app_mod.run_tool("calculer", {"expression": "(3+4)*2**3/7"}) == "8"
    assert "invalide" in app_mod.run_tool("calculer", {"expression": "__import__('os')"})
    assert "zéro" in app_mod.run_tool("calculer", {"expression": "1/0"})


def test_builtin_heure(app_mod):
    assert "UTC" in app_mod.run_tool("heure", {"fuseau": "UTC"})
    assert "inconnu" in app_mod.run_tool("heure", {"fuseau": "Nulle/Part"})


def test_builtin_meteo_mocked(app_mod, monkeypatch):
    class R:
        def __init__(self, d):
            self._d = d

        def json(self):
            return self._d

    def fake_get(url, **kw):
        if "geocoding" in url:
            return R({"results": [{"name": "Test", "country": "TC",
                                   "latitude": 1, "longitude": 2}]})
        return R({"current": {"temperature_2m": 20, "apparent_temperature": 19,
                              "relative_humidity_2m": 50, "wind_speed_10m": 5}})

    monkeypatch.setattr(app_mod.builtin_tools.requests, "get", fake_get)
    out = app_mod.run_tool("meteo", {"ville": "Test"})
    assert "Test" in out and "20" in out


def test_greffe_preserve_les_embarques(app_mod, monkeypatch):
    """Un outil greffé s'ajoute; les embarqués restent disponibles."""
    monkeypatch.setitem(app_mod.HIVE_TOOLS, "greffe_test", lambda: "ok")
    assert app_mod.run_tool("greffe_test", {}) == "ok"
    for name in app_mod.BUILTIN_TOOL_NAMES:
        assert name in app_mod.HIVE_TOOLS
