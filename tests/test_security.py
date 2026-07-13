"""Garde SSRF, validation d'URL et jeton d'accès."""


def test_url_scheme_invalide(app_mod):
    assert app_mod._url_blocked("ftp://example.com") is not None
    assert app_mod._url_blocked("file:///etc/passwd") is not None
    assert app_mod._url_blocked("") is not None
    assert app_mod._url_blocked(None) is not None


def test_url_privee_bloquee(app_mod):
    assert app_mod._url_blocked("http://127.0.0.1/admin") is not None
    assert app_mod._url_blocked("http://localhost:8080") is not None
    assert app_mod._url_blocked("http://169.254.169.254/latest/meta-data/") is not None
    assert app_mod._url_blocked("http://10.0.0.5/") is not None
    assert app_mod._url_blocked("http://192.168.1.1/") is not None


def test_url_publique_autorisee(app_mod):
    # IP littérale publique : pas de résolution DNS nécessaire
    assert app_mod._url_blocked("http://8.8.8.8/") is None


def test_host_irresoluble_traite_comme_prive(app_mod, monkeypatch):
    import socket
    def boom(*a, **k):
        raise socket.gaierror("nx")
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert app_mod._host_private("nexiste-pas.invalid") is True


def test_redirection_vers_ip_privee_bloquee(app_mod, monkeypatch):
    """Un serveur public qui redirige vers le réseau interne est bloqué
    AVANT que la requête du 2e saut ne parte."""
    calls = []

    class FakeResp:
        def __init__(self, url, redirect_to=None):
            self.url = url
            self.is_redirect = redirect_to is not None
            self.is_permanent_redirect = False
            self.headers = {"Location": redirect_to} if redirect_to else {}
            self.text = "contenu public"

    def fake_get(url, **kw):
        calls.append(url)
        if "8.8.8.8" in url:
            return FakeResp(url, redirect_to="http://169.254.169.254/secrets")
        return FakeResp(url)

    monkeypatch.setattr(app_mod.requests, "get", fake_get)
    out = app_mod._web_get("http://8.8.8.8/page")
    assert "bloquée" in out
    assert calls == ["http://8.8.8.8/page"]  # le saut interne n'est jamais requêté


def test_trop_de_redirections(app_mod, monkeypatch):
    class FakeResp:
        is_redirect = True
        is_permanent_redirect = False
        headers = {"Location": "http://8.8.8.8/loop"}
        text = ""

    monkeypatch.setattr(app_mod.requests, "get", lambda url, **kw: FakeResp())
    out = app_mod._web_get("http://8.8.8.8/loop")
    assert "redirections" in out


def test_check_token(app_mod):
    assert app_mod._check("test-token") is True
    assert app_mod._check("mauvais") is False
    assert app_mod._check("") is False


def test_ruche_task_refuse_sans_token(app_mod):
    assert "forbidden" in app_mod.ruche_task("fais un truc", token="mauvais")


def test_ruche_chat_refuse_sans_token(app_mod):
    assert "forbidden" in app_mod.ruche_chat("salut", token="")
