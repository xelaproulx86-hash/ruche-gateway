# ══════════════════════════════════════════════════════════════════
#  🐝 RUCHE-GATEWAY v3 — cascade économique + pont MCP
#  Base : v2.1 (revue 3-IA du 2026-07-09 matin, voir v2.1-original/)
#  Ajouts v3 :
#   - Cascade T1 GRATUIT (chaîne hivebase 'strong') → T2 GPU RunPod
#     (le GPU ne se réveille QUE si tier="gpu" demandé ou cascade épuisée)
#   - mcp_server=True → le même Space devient un serveur MCP
#     (pont Claude Code / claude.ai / Cowork : 1 URL pour les trois)
#   - gateway_info() : état des tiers + estimation de coût
#  Inchangé : SSRF guard, mémoire, n8n, greffe hivebase dormante,
#             portes /health /claude /hook, GATEWAY_TOKEN.
# ══════════════════════════════════════════════════════════════════
import os, json, time, socket, ipaddress, requests
from urllib.parse import urlparse
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

# ── Clés : self-load depuis ~/.hivebase-keys.env en mode LOCAL ──────
# (même patron que server.py d'assistant-web; sur un hébergeur cloud le
#  fichier n'existe pas et les secrets viennent de l'env — no-op)
def _ensure_keys():
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
        pass

_ensure_keys()

# ── Config (Settings > Variables and secrets du Space) ──────────────
RUNPOD_BASE_URL = os.environ.get("RUNPOD_BASE_URL", "").rstrip("/")  # https://api.runpod.ai/v2/<id>/openai/v1
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")                # secret
GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "change-me")         # secret
N8N_WEBHOOK = os.environ.get("N8N_WEBHOOK", "")                      # optionnel
MODEL = os.environ.get("MODEL_NAME", "qwen/qwen2.5-14b-instruct-awq")

MEM_FILE = "/tmp/ruche_memory.json"

# ── T1 : chaîne GRATUITE (copie fidèle de hivebase/config.py 'strong') ──
# Ordre = priorité. Un provider sans clé est sauté; un 429/403 le met en
# cooldown 90 s. Cerebras/Groq sont derrière Cloudflare → UA navigateur.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
FREE_CHAIN = [
    # (nom, base_url, modèle, env de la clé)
    ("sambanova", "https://api.sambanova.ai/v1", "Meta-Llama-3.3-70B-Instruct", "SAMBANOVA_API_KEY"),
    ("cerebras", "https://api.cerebras.ai/v1", "gpt-oss-120b", "CEREBRAS_API_KEY"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash-preview-04-17", "GEMINI_API_KEY"),
    ("mistral", "https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL_API_KEY"),
    ("nvidia", "https://integrate.api.nvidia.com/v1", "meta/llama-3.1-8b-instruct", "NVIDIA_API_KEY"),
    ("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
]
_COOLDOWN = {}  # provider -> timestamp de fin de pénalité


def _free_available():
    """Providers gratuits dont la clé est présente (dans l'ordre de la chaîne)."""
    return [p for p in FREE_CHAIN if os.environ.get(p[3])]


def _call_openai_compat(base_url, api_key, model, messages, max_tokens=800):
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": _UA},
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0.2},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm(messages, tier="auto", max_tokens=800):
    """Cascade économique. Retourne (texte, 'provider').
    tier='free' : gratuit seulement (erreur si tout est down).
    tier='gpu'  : RunPod direct (¢/s — réveille un worker).
    tier='auto' : gratuit d'abord, GPU en dernier recours."""
    errors = []
    if tier in ("free", "auto"):
        now = time.time()
        for name, base, model, key_env in _free_available():
            if _COOLDOWN.get(name, 0) > now:
                continue
            try:
                return _call_openai_compat(base, os.environ[key_env], model,
                                           messages, max_tokens), name
            except Exception as e:
                _COOLDOWN[name] = now + 90
                errors.append(f"{name}: {e}")
        if tier == "free":
            raise RuntimeError("chaîne gratuite épuisée — " + "; ".join(errors[-3:]))
    if tier in ("gpu", "auto"):
        if not (RUNPOD_BASE_URL and RUNPOD_API_KEY):
            raise RuntimeError("GPU non configuré (RUNPOD_BASE_URL/RUNPOD_API_KEY absents)"
                               + ("; gratuit épuisé: " + "; ".join(errors[-3:]) if errors else ""))
        return _call_openai_compat(RUNPOD_BASE_URL, RUNPOD_API_KEY, MODEL,
                                   messages, max_tokens), "runpod-gpu"
    raise RuntimeError(f"tier inconnu: {tier}")


# ── HIVEBASE: greffe des mains depuis GitHub (dormant sans GITHUB_TOKEN) ──
import subprocess, sys, inspect, importlib, pkgutil, shutil

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HIVE_REPO = os.environ.get("HIVE_REPO", "xelaproulx86-hash/ruche")
HIVE_MODULE = os.environ.get("HIVE_MODULE", "hivebase")
HIVE_AUTOSCAN = os.environ.get("HIVE_AUTOSCAN", "0") == "1"  # défaut: TOOLS explicite seulement

HIVE_TOOLS = {}  # name -> callable

def _safe_str(x, limit=3000):
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = str(x)
    return s[:limit]

def _clone_repo(dst="/tmp/hive"):
    """Clone en lecture via header Authorization (le token ne persiste pas dans .git/config)."""
    if not GITHUB_TOKEN:
        raise RuntimeError("pas de GITHUB_TOKEN")
    if not shutil.which("git"):
        raise RuntimeError("git introuvable dans l'image")
    subprocess.run(["rm", "-rf", dst], check=False)
    subprocess.run([
        "git",
        "-c", f"http.extraHeader=Authorization: Bearer {GITHUB_TOKEN}",
        "clone", "--depth", "1",
        f"https://github.com/{HIVE_REPO}.git",
        dst
    ], check=True)
    return dst

def _collect_public_functions(mod):
    tools = {}
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(fn, "__module__", "") != getattr(mod, "__name__", ""):
            continue
        tools[name] = fn
    return tools

def graft_hivebase():
    """Défaut: seul un dict TOOLS explicite est exposé. HIVE_AUTOSCAN=1 pour scanner le package."""
    global HIVE_TOOLS
    if not GITHUB_TOKEN:
        return "pas de GITHUB_TOKEN — mains locales seulement"
    dst = _clone_repo("/tmp/hive")
    if dst not in sys.path:
        sys.path.insert(0, dst)
    mod = importlib.import_module(HIVE_MODULE)
    if hasattr(mod, "TOOLS") and isinstance(getattr(mod, "TOOLS"), dict):
        HIVE_TOOLS = {k: v for k, v in mod.TOOLS.items() if callable(v) and not str(k).startswith("_")}
        return f"greffé (TOOLS): {', '.join(sorted(HIVE_TOOLS)) or 'aucune fonction'}"
    if not HIVE_AUTOSCAN:
        return "repo cloné, mais pas de dict TOOLS exporté — ajoute TOOLS = {\"nom\": fn} dans hivebase (ou HIVE_AUTOSCAN=1)"
    tools = _collect_public_functions(mod)
    if hasattr(mod, "__path__"):
        prefix = mod.__name__ + "."
        for _, name, _ in pkgutil.walk_packages(mod.__path__, prefix):
            try:
                sub = importlib.import_module(name)
                tools.update(_collect_public_functions(sub))
            except Exception:
                continue
    HIVE_TOOLS = tools
    return f"greffé (autoscan): {', '.join(sorted(HIVE_TOOLS)) or 'aucune fonction trouvée'}"

GRAFT_STATUS = ""
try:
    GRAFT_STATUS = graft_hivebase()
except Exception as e:
    GRAFT_STATUS = f"échec greffe: {e}"

# ── Garde anti-SSRF pour web_get ────────────────────────────────────
def _host_private(host):
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

def _url_blocked(url):
    p = urlparse(url or "")
    if p.scheme not in ("http", "https") or not p.hostname:
        return "URL invalide (http/https seulement)"
    if _host_private(p.hostname):
        return "URL bloquée (adresse interne/privée)"
    return None

# ── Mémoire simple (éphémère) ───────────────────────────────────────
def _mem():
    try:
        with open(MEM_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _mem_save(d):
    with open(MEM_FILE, "w") as f:
        json.dump(d, f)

# ── Prompt système (regénéré à chaque appel — reflète les greffes) ──
def _system_prompt():
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

def run_tool(action, args):
    try:
        if action in HIVE_TOOLS:
            fn = HIVE_TOOLS[action]
            if not isinstance(args, dict):
                args = {}
            return _safe_str(fn(**args), 3000)
        if action == "web_get":
            err = _url_blocked(args.get("url"))
            if err:
                return err
            rr = requests.get(args["url"], timeout=30,
                              headers={"User-Agent": "RucheGateway/1.0"})
            final_host = urlparse(rr.url).hostname
            if final_host and _host_private(final_host):
                return "URL bloquée après redirection (adresse interne/privée)"
            return rr.text[:3000]
        if action == "remember":
            m = _mem(); m[str(args["key"])] = str(args["value"]); _mem_save(m)
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
        return f"erreur outil: {e}"

# ── La boucle d'agent : penser -> agir -> observer -> recommencer ───
def _run_agent(task, tier="auto"):
    task = (task or "").strip()
    if not task:
        return "Tâche vide."
    log, providers = [], []
    msgs = [{"role": "system", "content": _system_prompt()},
            {"role": "user", "content": task}]
    try:
        for _ in range(5):
            raw, prov = llm(msgs, tier=tier)
            providers.append(prov)
            msgs.append({"role": "assistant", "content": raw})
            try:
                c = raw.strip()
                obj = json.loads(c[c.find("{"): c.rfind("}") + 1])
            except Exception:
                return raw
            action, args = obj.get("action"), obj.get("args", {})
            if action == "done":
                trail = " > ".join(log) if log else "réponse directe"
                return (f"{args.get('answer', '(fini)')}\n\n"
                        f"— actions: {trail} · cerveaux: {' > '.join(providers)}")
            obs = run_tool(action, args)
            log.append(action)
            msgs.append({"role": "user",
                         "content": f"Observation: {obs}\nContinue (JSON seulement)."})
        return "Limite de 5 actions atteinte.\n— actions: " + " > ".join(log)
    except Exception as e:
        return f"Erreur gateway: {e}"

def _check(token):
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
    try:
        text, prov = llm([{"role": "user", "content": prompt}], tier=tier)
        return f"{text}\n\n— cerveau: {prov}"
    except Exception as e:
        return f"Erreur: {e}"

def gateway_info() -> str:
    """État du gateway : tiers disponibles, greffe hivebase, coûts. Lecture seule, sans token."""
    free = [p[0] for p in _free_available()]
    return json.dumps({
        "tiers": {
            "free": free or "aucune clé gratuite configurée",
            "gpu": ("configuré — " + MODEL) if (RUNPOD_BASE_URL and RUNPOD_API_KEY)
                   else "non configuré",
        },
        "graft": GRAFT_STATUS,
        "hive_tools": sorted(HIVE_TOOLS.keys()),
        "coût": "free=0$ · gpu≈0.69$/h facturé à la seconde, 0$ au repos (scale-to-zero)",
    }, ensure_ascii=False, indent=2)

# ── Les portes HTTP historiques (compat v2.1 : téléphone, n8n, scripts) ──
app = FastAPI()

@app.get("/ping")
def ping():
    """Sonde santé requise par les endpoints load-balancing RunPod."""
    return {"status": "healthy"}

@app.get("/health")
def health():
    return {
        "ok": True,
        "brain": bool(RUNPOD_BASE_URL),
        "free_chain": [p[0] for p in _free_available()],
        "graft": GRAFT_STATUS,
        "hive_tools": sorted(HIVE_TOOLS.keys()),
    }

@app.get("/claude", response_class=PlainTextResponse)
def claude_gate(token: str = "", task: str = "", tier: str = "auto"):
    if not _check(token):
        return PlainTextResponse("forbidden", status_code=403)
    return _run_agent(task, tier=tier)

@app.post("/hook")
async def hook(req: Request):
    body = await req.json()
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
        t_tier = gr.Dropdown(["auto", "free", "gpu"], value="auto", label="Tier")
        t_token = gr.Textbox(label="Token", type="password")
        t_out = gr.Textbox(label="Résultat", lines=12)
        gr.Button("Lancer 🐝").click(ruche_task, [t_task, t_tier, t_token], t_out)
    with gr.Tab("Chat direct (1 appel)"):
        c_prompt = gr.Textbox(label="Prompt", lines=4)
        c_tier = gr.Dropdown(["free", "auto", "gpu"], value="free", label="Tier")
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
