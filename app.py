# ══════════════════════════════════════════════════════════════════
#  🐝 RUCHE-GATEWAY v3.1 — cascade économique + pont MCP
#  Base : v3 (revue 3-IA du 2026-07-09 matin, voir v2.1-original/)
#  Ajouts v3.1 :
#   - type hints partout + logging structuré (module `logging`)
#   - mémoire : écriture atomique (tmp + os.replace) sous verrou
#   - extraction JSON robuste (raw_decode, tolère le texte autour)
#   - garde SSRF : chaque saut de redirection est vérifié (pas
#     seulement l'URL finale)
#   - validation d'entrée : longueur max des tâches/prompts
#  Inchangé : cascade T1 gratuit → T2 GPU RunPod, greffe hivebase,
#             portes /ping /health /claude /hook, GATEWAY_TOKEN,
#             UI Gradio + mcp_server=True.
# ══════════════════════════════════════════════════════════════════
from __future__ import annotations

import concurrent.futures
import ipaddress
import json
import logging
import os
import socket
import tempfile
import threading
import time
from typing import Any, Callable, NamedTuple
from urllib.parse import urlparse

import gradio as gr
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("ruche")

# ── Clés : self-load depuis ~/.hivebase-keys.env en mode LOCAL ──────
# (même patron que server.py d'assistant-web; sur un hébergeur cloud le
#  fichier n'existe pas et les secrets viennent de l'env — no-op)
def _ensure_keys() -> None:
    import re
    from pathlib import Path
    stash = Path.home() / ".hivebase-keys.env"
    try:
        if stash.exists():
            for ln in stash.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^([A-Z0-9_]+)=(.+)$", ln.strip())
                if m:
                    os.environ.setdefault(m.group(1), m.group(2).strip())
    except Exception:
        log.warning("lecture de ~/.hivebase-keys.env impossible", exc_info=True)

_ensure_keys()

# ── Config (Settings > Variables and secrets du Space) ──────────────
RUNPOD_BASE_URL = os.environ.get("RUNPOD_BASE_URL", "").rstrip("/")  # https://api.runpod.ai/v2/<id>/openai/v1
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")                # secret
GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "change-me")         # secret
N8N_WEBHOOK = os.environ.get("N8N_WEBHOOK", "")                      # optionnel
MODEL = os.environ.get("MODEL_NAME", "qwen/qwen2.5-14b-instruct-awq")
MAX_TASK_CHARS = int(os.environ.get("MAX_TASK_CHARS", "8000"))

MEM_FILE = "/tmp/ruche_memory.json"

# ── T1 : chaîne GRATUITE (copie fidèle de hivebase/config.py 'strong') ──
# Ordre = priorité. Un provider sans clé est sauté; un échec le met en
# cooldown 90 s. Cerebras/Groq sont derrière Cloudflare → UA navigateur.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


class Provider(NamedTuple):
    name: str
    base_url: str
    model: str
    key_env: str


FREE_CHAIN: list[Provider] = [
    Provider("sambanova", "https://api.sambanova.ai/v1", "Meta-Llama-3.3-70B-Instruct", "SAMBANOVA_API_KEY"),
    Provider("cerebras", "https://api.cerebras.ai/v1", "gpt-oss-120b", "CEREBRAS_API_KEY"),
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash-preview-04-17", "GEMINI_API_KEY"),
    Provider("mistral", "https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL_API_KEY"),
    Provider("nvidia", "https://integrate.api.nvidia.com/v1", "meta/llama-3.1-8b-instruct", "NVIDIA_API_KEY"),
    Provider("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
]
COOLDOWN_SECONDS = 90.0
_COOLDOWN: dict[str, float] = {}  # provider -> timestamp de fin de pénalité
CONSEIL_ADVISORS = int(os.environ.get("CONSEIL_ADVISORS", "3"))  # nb de cerveaux consultés en parallèle


def _free_available() -> list[Provider]:
    """Providers gratuits dont la clé est présente (dans l'ordre de la chaîne)."""
    return [p for p in FREE_CHAIN if os.environ.get(p.key_env)]


def _call_openai_compat(base_url: str, api_key: str, model: str,
                        messages: list[dict[str, str]], max_tokens: int = 800) -> str:
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": _UA},
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.2},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm(messages: list[dict[str, str]], tier: str = "auto",
        max_tokens: int = 800) -> tuple[str, str]:
    """Cascade économique. Retourne (texte, 'provider').
    tier='free'    : gratuit seulement (erreur si tout est down).
    tier='gpu'     : RunPod direct (¢/s — réveille un worker).
    tier='auto'    : gratuit d'abord, GPU en dernier recours.
    tier='conseil' : N cerveaux gratuits en parallèle + 1 arbitre (0$)."""
    if tier == "conseil":
        return _llm_conseil(messages, max_tokens)
    errors: list[str] = []
    if tier in ("free", "auto"):
        now = time.time()
        for prov in _free_available():
            if _COOLDOWN.get(prov.name, 0) > now:
                continue
            try:
                text = _call_openai_compat(prov.base_url, os.environ[prov.key_env],
                                           prov.model, messages, max_tokens)
                return text, prov.name
            except Exception as e:
                _COOLDOWN[prov.name] = now + COOLDOWN_SECONDS
                errors.append(f"{prov.name}: {e}")
                log.warning("provider %s en échec, cooldown %ss: %s",
                            prov.name, COOLDOWN_SECONDS, e)
        if tier == "free":
            raise RuntimeError("chaîne gratuite épuisée — " + "; ".join(errors[-3:]))
    if tier in ("gpu", "auto"):
        if not (RUNPOD_BASE_URL and RUNPOD_API_KEY):
            raise RuntimeError("GPU non configuré (RUNPOD_BASE_URL/RUNPOD_API_KEY absents)"
                               + ("; gratuit épuisé: " + "; ".join(errors[-3:]) if errors else ""))
        return _call_openai_compat(RUNPOD_BASE_URL, RUNPOD_API_KEY, MODEL,
                                   messages, max_tokens), "runpod-gpu"
    raise RuntimeError(f"tier inconnu: {tier}")


def _llm_conseil(messages: list[dict[str, str]], max_tokens: int = 800) -> tuple[str, str]:
    """Mode conseil (« tri-brain ») : interroge plusieurs cerveaux gratuits EN
    PARALLÈLE, puis un arbitre synthétise/tranche. Coût 0$. Repli sur la
    cascade 'auto' si moins de 2 conseillers sont disponibles ou répondent."""
    now = time.time()
    avail = [p for p in _free_available() if _COOLDOWN.get(p.name, 0) <= now]
    advisors = avail[:CONSEIL_ADVISORS]
    if len(advisors) < 2:
        return llm(messages, tier="auto", max_tokens=max_tokens)

    opinions: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(advisors)) as ex:
        futs = {ex.submit(_call_openai_compat, p.base_url, os.environ[p.key_env],
                          p.model, messages, max_tokens): p for p in advisors}
        for fut in concurrent.futures.as_completed(futs):
            p = futs[fut]
            try:
                opinions.append((p.name, fut.result()))
            except Exception as e:
                _COOLDOWN[p.name] = now + COOLDOWN_SECONDS
                log.warning("conseil: %s en échec: %s", p.name, e)

    if not opinions:
        return llm(messages, tier="auto", max_tokens=max_tokens)
    if len(opinions) == 1:
        name, text = opinions[0]
        return text, f"conseil→{name} (seul répondant)"

    # Arbitre : un provider frais (hors conseillers si possible) synthétise.
    consulted = {n for n, _ in opinions}
    arbiter = next((p for p in avail if p.name not in consulted), advisors[0])
    joined = "\n\n".join(f"--- Réponse {i + 1} ({name}) ---\n{txt}"
                         for i, (name, txt) in enumerate(opinions))
    arb_messages = [
        {"role": "system", "content":
            "Tu es l'arbitre d'un conseil de plusieurs assistants IA. On te fournit la "
            "requête d'origine puis plusieurs réponses indépendantes. Produis UNE seule "
            "réponse consolidée, la plus juste et complète possible : corrige les erreurs, "
            "tranche les désaccords, garde le meilleur de chacune. Conserve EXACTEMENT le "
            "format attendu des réponses (si elles sont en JSON, réponds en JSON valide et "
            "rien d'autre)."},
        {"role": "user", "content":
            f"Requête d'origine :\n{messages[-1]['content']}\n\n"
            f"Réponses du conseil :\n{joined}\n\nRéponse consolidée :"},
    ]
    names = "+".join(n for n, _ in opinions)
    try:
        synth = _call_openai_compat(arbiter.base_url, os.environ[arbiter.key_env],
                                    arbiter.model, arb_messages, max_tokens)
        return synth, f"conseil({names})→{arbiter.name}"
    except Exception as e:
        log.warning("conseil: arbitre %s en échec: %s", arbiter.name, e)
        best_name, best_text = max(opinions, key=lambda o: len(o[1]))
        return best_text, f"conseil({names})→{best_name} (arbitre indispo)"


# ── HIVEBASE: greffe des mains depuis GitHub (dormant sans GITHUB_TOKEN) ──
import importlib
import inspect
import pkgutil
import shutil
import subprocess
import sys

import builtin_tools

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HIVE_REPO = os.environ.get("HIVE_REPO", "xelaproulx86-hash/ruche")
HIVE_MODULE = os.environ.get("HIVE_MODULE", "hivebase")
HIVE_AUTOSCAN = os.environ.get("HIVE_AUTOSCAN", "0") == "1"  # défaut: TOOLS explicite seulement

# Outils EMBARQUÉS (vendored, cuits dans l'image) : toujours présents dès le
# démarrage, sans token ni réseau. La greffe GitHub s'ajoute/écrase par-dessus.
HIVE_TOOLS: dict[str, Callable[..., Any]] = {
    k: v for k, v in builtin_tools.TOOLS.items() if callable(v)
}
BUILTIN_TOOL_NAMES = sorted(HIVE_TOOLS)


def _safe_str(x: Any, limit: int = 3000) -> str:
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = str(x)
    return s[:limit]


def _clone_repo(dst: str = "/tmp/hive") -> str:
    """Clone en lecture. Le token passe par les variables GIT_CONFIG_* :
    jamais dans argv (visible via ps / messages d'erreur), jamais dans
    .git/config. L'erreur levée est assainie — GRAFT_STATUS est affiché
    publiquement dans /health et l'UI."""
    if not GITHUB_TOKEN:
        raise RuntimeError("pas de GITHUB_TOKEN")
    if not shutil.which("git"):
        raise RuntimeError("git introuvable dans l'image")
    shutil.rmtree(dst, ignore_errors=True)
    env = dict(
        os.environ,
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.extraHeader",
        GIT_CONFIG_VALUE_0=f"Authorization: Bearer {GITHUB_TOKEN}",
    )
    proc = subprocess.run(
        ["git", "clone", "--depth", "1",
         f"https://github.com/{HIVE_REPO}.git", dst],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.warning("clone hivebase en échec (git exit %s): %s",
                    proc.returncode, proc.stderr.strip())
        raise RuntimeError(f"clone de {HIVE_REPO} en échec (git exit {proc.returncode})")
    return dst


def _collect_public_functions(mod: Any) -> dict[str, Callable[..., Any]]:
    tools: dict[str, Callable[..., Any]] = {}
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(fn, "__module__", "") != getattr(mod, "__name__", ""):
            continue
        tools[name] = fn
    return tools


def graft_hivebase() -> str:
    """Défaut: seul un dict TOOLS explicite est exposé. HIVE_AUTOSCAN=1 pour scanner le package."""
    if not GITHUB_TOKEN:
        return f"pas de GITHUB_TOKEN — {len(BUILTIN_TOOL_NAMES)} outils embarqués seulement ({', '.join(BUILTIN_TOOL_NAMES)})"
    dst = _clone_repo("/tmp/hive")
    if dst not in sys.path:
        sys.path.insert(0, dst)
    mod = importlib.import_module(HIVE_MODULE)
    if hasattr(mod, "TOOLS") and isinstance(getattr(mod, "TOOLS"), dict):
        grafted = {k: v for k, v in mod.TOOLS.items() if callable(v) and not str(k).startswith("_")}
        HIVE_TOOLS.update(grafted)  # s'ajoute aux embarqués
        return f"greffé (TOOLS): {', '.join(sorted(grafted)) or 'aucune fonction'} (+ {len(BUILTIN_TOOL_NAMES)} embarqués)"
    if not HIVE_AUTOSCAN:
        return f"repo cloné, mais pas de dict TOOLS exporté — {len(BUILTIN_TOOL_NAMES)} outils embarqués actifs (ajoute TOOLS = {{\"nom\": fn}} dans hivebase, ou HIVE_AUTOSCAN=1)"
    tools = _collect_public_functions(mod)
    if hasattr(mod, "__path__"):
        prefix = mod.__name__ + "."
        for _, name, _ in pkgutil.walk_packages(mod.__path__, prefix):
            try:
                sub = importlib.import_module(name)
                tools.update(_collect_public_functions(sub))
            except Exception:
                log.warning("greffe: import de %s impossible", name, exc_info=True)
                continue
    HIVE_TOOLS.update(tools)  # s'ajoute aux embarqués
    return f"greffé (autoscan): {', '.join(sorted(tools)) or 'aucune fonction trouvée'} (+ {len(BUILTIN_TOOL_NAMES)} embarqués)"


GRAFT_STATUS = ""
try:
    GRAFT_STATUS = graft_hivebase()
except Exception as e:
    GRAFT_STATUS = f"échec greffe: {e}"
    log.warning("échec de la greffe hivebase", exc_info=True)
log.info("greffe hivebase: %s", GRAFT_STATUS)

# ── Garde anti-SSRF pour web_get ────────────────────────────────────
def _host_private(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return True
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except Exception:
            return True
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return True
    return False


def _url_blocked(url: str | None) -> str | None:
    p = urlparse(url or "")
    if p.scheme not in ("http", "https") or not p.hostname:
        return "URL invalide (http/https seulement)"
    if _host_private(p.hostname):
        return "URL bloquée (adresse interne/privée)"
    return None


def _web_get(url: str, max_redirects: int = 5) -> str:
    """GET avec redirections suivies manuellement : CHAQUE saut passe la
    garde SSRF (pas seulement l'URL finale)."""
    for _ in range(max_redirects + 1):
        err = _url_blocked(url)
        if err:
            return err
        rr = requests.get(url, timeout=30, allow_redirects=False,
                          headers={"User-Agent": "RucheGateway/1.0"})
        if rr.is_redirect or rr.is_permanent_redirect:
            nxt = rr.headers.get("Location", "")
            url = requests.compat.urljoin(url, nxt)
            continue
        return rr.text[:3000]
    return "URL bloquée (trop de redirections)"

# ── Vault : persistance git OPTIONNELLE de la mémoire ───────────────
# Sans VAULT_REPO (ou sans GITHUB_TOKEN), rien ne change : mémoire
# éphémère dans /tmp. Avec, la mémoire vit dans un clone du vault et
# chaque `remember` est commité + poussé (best-effort).
VAULT_REPO = os.environ.get("VAULT_REPO", "")   # ex: xelaproulx86-hash/ruche-memory-vault
VAULT_DIR = os.environ.get("VAULT_DIR", "/tmp/ruche_vault")
_VAULT_ACTIVE = False


def _git_auth_env() -> dict[str, str]:
    """Token via GIT_CONFIG_* : jamais dans argv (ps, messages d'erreur)."""
    return dict(
        os.environ,
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.extraHeader",
        GIT_CONFIG_VALUE_0=f"Authorization: Bearer {GITHUB_TOKEN}",
    )


def _vault_git(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(["git", "-C", VAULT_DIR, *args],
                          env=_git_auth_env(), capture_output=True, text=True)


def vault_init() -> str:
    """Clone le vault et y déplace MEM_FILE. Erreurs assainies (le statut
    est affiché publiquement dans /health)."""
    global _VAULT_ACTIVE, MEM_FILE
    if not VAULT_REPO:
        return "vault désactivé (VAULT_REPO absent) — mémoire éphémère /tmp"
    if not GITHUB_TOKEN:
        return "vault désactivé (GITHUB_TOKEN absent) — mémoire éphémère /tmp"
    if not shutil.which("git"):
        return "vault désactivé (git introuvable) — mémoire éphémère /tmp"
    shutil.rmtree(VAULT_DIR, ignore_errors=True)
    proc = subprocess.run(
        ["git", "clone", "--depth", "1",
         f"https://github.com/{VAULT_REPO}.git", VAULT_DIR],
        env=_git_auth_env(), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.warning("clone du vault en échec (git exit %s): %s",
                    proc.returncode, proc.stderr.strip())
        return f"vault en échec (git exit {proc.returncode}) — repli mémoire /tmp"
    _vault_git("config", "user.email", "ruche-gateway@laruche.local")
    _vault_git("config", "user.name", "ruche-gateway")
    MEM_FILE = os.path.join(VAULT_DIR, "memory.json")
    _VAULT_ACTIVE = True
    return f"vault actif ({VAULT_REPO})"


def _vault_push() -> None:
    """Commit + push best-effort après une écriture (jamais bloquant :
    un échec réseau laisse la copie locale intacte et se rattrape au
    push suivant)."""
    if not _VAULT_ACTIVE:
        return
    _vault_git("add", "memory.json")
    c = _vault_git("commit", "-m", "ruche: mémoire mise à jour")
    if c.returncode != 0:
        return  # rien de nouveau à committer
    p = _vault_git("push")
    if p.returncode != 0:
        log.warning("push du vault en échec: %s", p.stderr.strip())


# ── Mémoire simple (vault si configuré, sinon éphémère) ─────────────
_MEM_LOCK = threading.Lock()


def _mem() -> dict[str, Any]:
    try:
        with open(MEM_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _mem_save(d: dict[str, Any]) -> None:
    """Écriture atomique : fichier temporaire puis os.replace (pas de
    fichier à moitié écrit si deux requêtes se chevauchent)."""
    dirname = os.path.dirname(MEM_FILE) or "."
    fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".ruche_mem_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(d, f)
        os.replace(tmp, MEM_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _mem_set(key: str, value: str) -> None:
    with _MEM_LOCK:
        m = _mem()
        m[key] = value
        _mem_save(m)
        _vault_push()


VAULT_STATUS = vault_init()
log.info("vault: %s", VAULT_STATUS)

# ── Prompt système (regénéré à chaque appel — reflète les greffes) ──
def _system_prompt() -> str:
    hive_names = ", ".join(sorted(HIVE_TOOLS.keys()))
    hive_line = ""
    if hive_names:
        hive_line = f"\n- greffées (hivebase) : {hive_names} -> appelle EXACTEMENT le nom (action=\"<nom>\") avec ses paramètres dans args"
    return f"""Tu es RUCHE-GATEWAY, l'orchestrateur exécuteur de La Ruche.
À chaque tour, réponds UNIQUEMENT avec un objet JSON, rien d'autre :
{{"action": "<nom>", "args": {{...}}}}

Actions disponibles :
- web_get : {{"url": "https://..."}} -> lit une page web (texte)
- remember : {{"key": "...", "value": "..."}} -> mémorise une info
- recall : {{"key": "..."}} -> relit la mémoire
- n8n : {{"payload": {{...}}}} -> déclenche le webhook n8n
- done : {{"answer": "..."}} -> termine, livre la réponse finale{hive_line}

Maximum 5 actions. Si aucune action n'est nécessaire, réponds directement avec done.
"""


def run_tool(action: str, args: Any) -> str:
    try:
        if action in HIVE_TOOLS:
            fn = HIVE_TOOLS[action]
            if not isinstance(args, dict):
                args = {}
            return _safe_str(fn(**args), 3000)
        if action == "web_get":
            return _web_get(args.get("url"))
        if action == "remember":
            _mem_set(str(args["key"]), str(args["value"]))
            return "mémorisé"
        if action == "recall":
            return str(_mem().get(str(args["key"]), "(vide)"))
        if action == "n8n":
            if not N8N_WEBHOOK:
                return "n8n non configuré (variable N8N_WEBHOOK absente)"
            rr = requests.post(N8N_WEBHOOK, json=args.get("payload", {}), timeout=30)
            return f"n8n -> HTTP {rr.status_code}"
        return f"action inconnue: {action}"
    except TypeError as e:
        return f"erreur args pour {action}: {e}"
    except Exception as e:
        log.warning("outil %s en erreur: %s", action, e)
        return f"erreur outil: {e}"

# ── La boucle d'agent : penser -> agir -> observer -> recommencer ───
def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extrait le premier objet JSON d'une réponse LLM, en tolérant du
    texte avant/après (prose, ```json, plusieurs objets…)."""
    c = raw.strip()
    try:
        obj = json.loads(c)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    dec = json.JSONDecoder()
    idx = c.find("{")
    while idx != -1:
        try:
            obj, _ = dec.raw_decode(c, idx)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        idx = c.find("{", idx + 1)
    return None


def _run_agent(task: str, tier: str = "auto") -> str:
    task = (task or "").strip()
    if not task:
        return "Tâche vide."
    if len(task) > MAX_TASK_CHARS:
        return f"Tâche trop longue ({len(task)} caractères, max {MAX_TASK_CHARS})."
    log_actions: list[str] = []
    providers: list[str] = []
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": task},
    ]
    try:
        for _ in range(5):
            raw, prov = llm(msgs, tier=tier)
            providers.append(prov)
            msgs.append({"role": "assistant", "content": raw})
            obj = _extract_json(raw)
            if obj is None:
                return raw
            action, args = obj.get("action"), obj.get("args", {})
            if action == "done":
                trail = " > ".join(log_actions) if log_actions else "réponse directe"
                return (f"{args.get('answer', '(fini)')}\n\n"
                        f"— actions: {trail} · cerveaux: {' > '.join(providers)}")
            obs = run_tool(action, args)
            log_actions.append(action)
            msgs.append({"role": "user",
                         "content": f"Observation: {obs}\nContinue (JSON seulement)."})
        return "Limite de 5 actions atteinte.\n— actions: " + " > ".join(log_actions)
    except Exception as e:
        log.warning("boucle agent en erreur: %s", e)
        return f"Erreur gateway: {e}"


def _check(token: str) -> bool:
    return token == GATEWAY_TOKEN

# ── Fonctions exposées (UI Gradio + tools MCP) ──────────────────────
def ruche_task(task: str, tier: str = "auto", token: str = "") -> str:
    """Confie une tâche à l'agent RUCHE (boucle penser→agir→observer, max 5 actions).

    Args:
        task: la tâche à accomplir (ex: "lis https://example.com et résume en 3 points")
        tier: "auto" (gratuit d'abord, GPU en dernier recours), "free" (gratuit seulement, 0$), "gpu" (RunPod direct, ¢/s)
        token: jeton du gateway (GATEWAY_TOKEN) — requis
    """
    if not _check(token):
        return "forbidden — fournis le GATEWAY_TOKEN dans le paramètre token"
    return _run_agent(task, tier=tier)


def ruche_chat(prompt: str, tier: str = "free", token: str = "") -> str:
    """Complétion directe SANS boucle d'agent (1 seul appel LLM — le plus économe).

    Args:
        prompt: la question/instruction
        tier: "free" (défaut, 0$), "auto", ou "gpu" (RunPod, ¢/s)
        token: jeton du gateway (GATEWAY_TOKEN) — requis
    """
    if not _check(token):
        return "forbidden — fournis le GATEWAY_TOKEN dans le paramètre token"
    prompt = (prompt or "").strip()
    if not prompt:
        return "Prompt vide."
    if len(prompt) > MAX_TASK_CHARS:
        return f"Prompt trop long ({len(prompt)} caractères, max {MAX_TASK_CHARS})."
    try:
        text, prov = llm([{"role": "user", "content": prompt}], tier=tier)
        return f"{text}\n\n— cerveau: {prov}"
    except Exception as e:
        return f"Erreur: {e}"


def gateway_info() -> str:
    """État du gateway : tiers disponibles, greffe hivebase, coûts. Lecture seule, sans token."""
    free = [p.name for p in _free_available()]
    return json.dumps({
        "tiers": {
            "free": free or "aucune clé gratuite configurée",
            "gpu": ("configuré — " + MODEL) if (RUNPOD_BASE_URL and RUNPOD_API_KEY)
                   else "non configuré",
        },
        "graft": GRAFT_STATUS,
        "hive_tools": sorted(HIVE_TOOLS.keys()),
        "builtin_tools": BUILTIN_TOOL_NAMES,
        "vault": VAULT_STATUS,
        "conseil": f"{CONSEIL_ADVISORS} cerveaux gratuits en parallèle + arbitre (tier=conseil)",
        "coût": "free/conseil=0$ · gpu≈0.69$/h facturé à la seconde, 0$ au repos (scale-to-zero)",
    }, ensure_ascii=False, indent=2)

# ── Les portes HTTP historiques (compat v2.1 : téléphone, n8n, scripts) ──
app = FastAPI()


@app.get("/ping")
def ping() -> dict[str, str]:
    """Sonde santé requise par les endpoints load-balancing RunPod."""
    return {"status": "healthy"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "brain": bool(RUNPOD_BASE_URL),
        "free_chain": [p.name for p in _free_available()],
        "graft": GRAFT_STATUS,
        "hive_tools": sorted(HIVE_TOOLS.keys()),
        "builtin_tools": BUILTIN_TOOL_NAMES,
        "vault": VAULT_STATUS,
    }


@app.get("/claude", response_class=PlainTextResponse)
def claude_gate(token: str = "", task: str = "", tier: str = "auto"):
    if not _check(token):
        return PlainTextResponse("forbidden", status_code=403)
    return _run_agent(task, tier=tier)


@app.post("/hook")
async def hook(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "corps JSON invalide"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "corps JSON invalide"}, status_code=400)
    if not _check(body.get("token", "")):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return {"result": _run_agent(body.get("task", ""), tier=body.get("tier", "auto"))}

# ── UI Gradio + serveur MCP (le pont : 1 URL pour les 3 surfaces Claude) ──
with gr.Blocks(title="🐝 RUCHE-GATEWAY v3") as demo:
    gr.Markdown(f"# 🐝 RUCHE-GATEWAY v3\n"
                f"Cascade : gratuit → GPU RunPod (Qwen 14B AWQ, 24 Go, Islande). "
                f"Greffe : {GRAFT_STATUS}\n\n"
                f"MCP : `/gradio_api/mcp/` · Portes : `GET /claude` · `POST /hook` · `GET /health`")
    with gr.Tab("Tâche (agent)"):
        t_task = gr.Textbox(label="Tâche", lines=4,
                            placeholder="Ex: Va lire https://example.com et résume en 3 points.")
        t_tier = gr.Dropdown(["auto", "free", "conseil", "gpu"], value="auto", label="Tier")
        t_token = gr.Textbox(label="Token", type="password")
        t_out = gr.Textbox(label="Résultat", lines=12)
        gr.Button("Lancer 🐝").click(ruche_task, [t_task, t_tier, t_token], t_out)
    with gr.Tab("Chat direct (1 appel)"):
        c_prompt = gr.Textbox(label="Prompt", lines=4)
        c_tier = gr.Dropdown(["free", "conseil", "auto", "gpu"], value="free", label="Tier")
        c_token = gr.Textbox(label="Token", type="password")
        c_out = gr.Textbox(label="Réponse", lines=12)
        gr.Button("Envoyer").click(ruche_chat, [c_prompt, c_tier, c_token], c_out)
    with gr.Tab("État"):
        i_out = gr.Textbox(label="gateway_info", lines=12)
        gr.Button("Rafraîchir").click(gateway_info, [], i_out)

app = gr.mount_gradio_app(app, demo, path="/", mcp_server=True)

if __name__ == "__main__":
    # local: 7860 · RunPod load-balancer: injecte PORT
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
