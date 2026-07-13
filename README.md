# 🐝 ruche-gateway

Gateway La Ruche : cascade economique (gratuit -> GPU RunPod) + serveur MCP.
Image CPU pour endpoint RunPod load-balancing. Aucune cle dans le code (env only).

## Architecture

```
RUCHE-GATEWAY v3.1
├─ T1: cascade GRATUITE (SambaNova, Cerebras, Gemini, Mistral, NVIDIA, Groq)
│      un provider sans cle est saute; un echec = cooldown 90 s
├─ T2: GPU RunPod (scale-to-zero — ne se reveille que si demande/necessaire)
├─ Greffe hivebase : charge des outils depuis un repo GitHub (dormante sans GITHUB_TOKEN)
├─ Garde SSRF : validation d'URL + verification de CHAQUE saut de redirection
├─ Memoire ephemere : /tmp/ruche_memory.json (ecriture atomique, verrou)
├─ FastAPI : GET /ping · GET /health · GET /claude · POST /hook
├─ UI Gradio : 3 onglets (Tache agent, Chat direct, Etat)
└─ Serveur MCP : /gradio_api/mcp/ (pont Claude Code / claude.ai / Cowork)
```

## Demarrage

```bash
pip install -r requirements.txt
cp .env.example .env   # puis remplis au moins GATEWAY_TOKEN + une cle gratuite
python app.py          # http://localhost:7860
```

Ou via Docker (image CPU pour endpoint RunPod load-balancing) :

```bash
docker build -t ruche-gateway .
docker run -p 7860:7860 -e GATEWAY_TOKEN=mon-jeton ruche-gateway
```

## Configuration

Toutes les variables sont documentees dans [`.env.example`](.env.example).
L'essentiel :

| Variable | Role |
| -------- | ---- |
| `GATEWAY_TOKEN` | jeton requis par les portes protegees (defaut `change-me` — change-le !) |
| `SAMBANOVA_API_KEY`… | cles T1 gratuites — chacune optionnelle, l'ordre de la chaine = la priorite |
| `RUNPOD_BASE_URL` / `RUNPOD_API_KEY` | T2 GPU (optionnel) |
| `MODEL_NAME` | modele servi par RunPod (defaut Qwen 2.5 14B AWQ) |
| `GITHUB_TOKEN` / `HIVE_REPO` | greffe d'outils hivebase (dormante sans token) |
| `MAX_TASK_CHARS` | longueur max d'une tache/prompt (defaut 8000) |

## API

Tiers : `free` (gratuit seulement, 0$) · `gpu` (RunPod direct, ¢/s) ·
`auto` (gratuit d'abord, GPU en dernier recours).

| Porte | Auth | Description |
| ----- | ---- | ----------- |
| `GET /ping` | non | sonde sante RunPod load-balancing |
| `GET /health` | non | etat : tiers, greffe, outils |
| `GET /claude?token=…&task=…&tier=auto` | token | boucle d'agent (max 5 actions), texte brut |
| `POST /hook` `{"token": …, "task": …, "tier": …}` | token | idem, JSON |
| `/gradio_api/mcp/` | token (par outil) | serveur MCP : `ruche_task`, `ruche_chat`, `gateway_info` |

L'agent dispose des actions `web_get` (avec garde SSRF), `remember` /
`recall` (memoire ephemere), `n8n` (webhook) et des outils greffes hivebase.

## Tests

```bash
pip install pytest
pytest
```

La suite (`tests/`) couvre la cascade LLM (ordre, cooldown, epuisement),
la garde SSRF (schemas, adresses privees, redirections), la memoire
(ecriture atomique, acces concurrents), les outils et la boucle d'agent
(extraction JSON, limites). Aucun test ne fait de vrai appel reseau.
La CI GitHub Actions ([.github/workflows/test.yml](.github/workflows/test.yml))
la lance sur chaque push/PR.
