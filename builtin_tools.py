# ══════════════════════════════════════════════════════════════════
#  🐝 builtin_tools — les mains EMBARQUÉES de RUCHE-GATEWAY
#  Copie vendue (vendored) des outils hivebase, cuite directement dans
#  l'image. Toujours présentes, dès le démarrage, sans GITHUB_TOKEN ni
#  réseau. La greffe hivebase (si GITHUB_TOKEN) vient s'ajouter/écraser
#  par-dessus ces outils de base.
#  Contraintes gateway : stdlib + requests seulement, résultat tronqué
#  à 3000 caractères, args issus du JSON du LLM (donc des chaînes).
# ══════════════════════════════════════════════════════════════════
from __future__ import annotations

import ast
import datetime
import operator
import zoneinfo

import requests

_TIMEOUT = 15


def heure(fuseau: str = "America/Toronto") -> str:
    """Date et heure actuelles dans un fuseau IANA (défaut: America/Toronto)."""
    try:
        tz = zoneinfo.ZoneInfo(fuseau)
    except Exception:
        return f"fuseau inconnu: {fuseau} (ex: America/Toronto, Europe/Paris, UTC)"
    return datetime.datetime.now(tz).strftime(f"%Y-%m-%d %H:%M:%S ({fuseau})")


# Évaluation arithmétique SÛRE : AST seulement, pas d'eval().
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ValueError("exposant trop grand")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"expression non autorisée: {ast.dump(node)[:60]}")


def calculer(expression: str) -> str:
    """Calcule une expression arithmétique (+ - * / // % ** et parenthèses)."""
    try:
        result = _eval_node(ast.parse(str(expression), mode="eval").body)
    except ZeroDivisionError:
        return "division par zéro"
    except (ValueError, SyntaxError) as e:
        return f"expression invalide: {e}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def meteo(ville: str) -> str:
    """Météo actuelle d'une ville (open-meteo, gratuit, sans clé)."""
    g = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": str(ville), "count": 1, "language": "fr"},
        timeout=_TIMEOUT,
    ).json()
    if not g.get("results"):
        return f"ville introuvable: {ville}"
    lieu = g["results"][0]
    w = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lieu["latitude"], "longitude": lieu["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m"},
        timeout=_TIMEOUT,
    ).json()
    cur = w.get("current", {})
    if not cur:
        return f"météo indisponible pour {ville}"
    pays = lieu.get("country", "")
    return (f"{lieu['name']}{', ' + pays if pays else ''} : "
            f"{cur.get('temperature_2m')}°C "
            f"(ressenti {cur.get('apparent_temperature')}°C), "
            f"humidité {cur.get('relative_humidity_2m')}%, "
            f"vent {cur.get('wind_speed_10m')} km/h")


# Le contrat : un dict TOOLS explicite, comme hivebase.
TOOLS = {
    "heure": heure,
    "calculer": calculer,
    "meteo": meteo,
}
