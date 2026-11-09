"""Métricas de monitoreo calculadas sobre el registro de uso.

Tres preguntas que el comité necesita poder responder en cualquier momento:

1. ¿El servicio funciona?           volumen, latencia p95, errores de entrada.
2. ¿El mundo cambió?                 deriva (PSI) del puntaje y de las entradas
                                     contra la distribución de validación.
3. ¿El modelo sigue acertando?       con las etiquetas que ya llegaron: recall,
                                     precisión y costo realizado del campeón y del
                                     retador sobre el MISMO tráfico.

PSI (Population Stability Index): sum((actual - esperado) * ln(actual / esperado))
sobre los deciles de validación. Regla usual de la industria: < 0.10 estable,
0.10–0.25 vigilar, > 0.25 cambio importante.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-4
PSI_VIGILAR, PSI_ALERTA = 0.10, 0.25
LATENCIA_OBJETIVO_MS = 100.0
MIN_PARA_PSI = 100


def psi(valores: np.ndarray, cortes: list[float], esperado: list[float]) -> tuple[float, list[float]]:
    cortes = np.asarray(cortes, dtype=float)
    internos = cortes[1:-1]
    idx = np.searchsorted(internos, valores, side="right")
    actual = np.bincount(idx, minlength=len(esperado))[: len(esperado)] / max(len(valores), 1)
    e = np.clip(np.asarray(esperado), EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e))), actual.tolist()


def psi_categorico(valores: list[str], esperado: dict[str, float]) -> float:
    n = max(len(valores), 1)
    total = 0.0
    for cat in set(esperado) | set(valores):
        a = max(valores.count(cat) / n, EPS)
        e = max(esperado.get(cat, 0.0), EPS)
        total += (a - e) * math.log(a / e)
    return float(total)


def _nivel_psi(v: float | None) -> str:
    if v is None:
        return "sin_datos"
    return "alerta" if v > PSI_ALERTA else "vigilar" if v > PSI_VIGILAR else "ok"


def _desempeno(y: np.ndarray, p: np.ndarray, decision: np.ndarray, costos: dict) -> dict:
    bloquea = decision == "block"
    tp = int((bloquea & (y == 1)).sum())
    fp = int((bloquea & (y == 0)).sum())
    fn = int((~bloquea & (y == 1)).sum())
    revisa = decision == "review"
    return {
        "n": int(len(y)), "n_fraude": int(y.sum()),
        "recall_bloqueo": tp / (tp + fn) if tp + fn else None,
        "precision_bloqueo": tp / (tp + fp) if tp + fp else None,
        "fraudes_en_revision": int((revisa & (y == 1)).sum()),
        "auc_pr": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None,
        "tp": tp, "fp": fp, "fn": fn,
        "costo_Q": fn * costos["FN"] + fp * costos["FP"],
    }


def _serie(filas: list[dict], n_tramos: int = 24) -> list[dict]:
    if not filas:
        return []
    ts = np.array([np.datetime64(f["recibido_utc"].replace("+00:00", "")) for f in filas])
    t0, t1 = ts.min(), ts.max()
    paso = max((t1 - t0) / n_tramos, np.timedelta64(1, "s"))
    tramo = ((ts - t0) / paso).astype(int).clip(0, n_tramos - 1)
    salida = []
    for i in range(int(tramo.max()) + 1):
        sel = [f for f, t in zip(filas, tramo) if t == i]
        if not sel:
            continue
        lat = [f["latencia_ms"] for f in sel]
        salida.append({
            "desde": str(t0 + paso * i)[:19],
            "n": len(sel),
            "block": sum(f["decision"] == "block" for f in sel),
            "review": sum(f["decision"] == "review" for f in sel),
            "latencia_p95": float(np.percentile(lat, 95)),
        })
    return salida


def resumen(filas: list[dict], referencia: dict, meta: dict) -> dict:
    roles = meta["roles"]
    costos = meta["costos"]
    salida: dict = {
        "version_modelo": meta["version_modelo"],
        "campeon": roles["campeon"]["familia"],
        "retador": roles["retador"]["familia"],
        "n_predicciones": len(filas),
        "alertas": [],
    }
    if not filas:
        salida["alertas"].append({"nivel": "info", "mensaje": "Aún no hay tráfico registrado."})
        return salida

    dec = np.array([f["decision"] for f in filas])
    dec_r = np.array([f["decision_retador"] for f in filas])
    lat = np.array([f["latencia_ms"] for f in filas], dtype=float)
    salida["decisiones"] = {d: float((dec == d).mean()) for d in ("allow", "review", "block")}
    salida["decisiones_retador"] = {d: float((dec_r == d).mean()) for d in ("allow", "review", "block")}
    salida["acuerdo_campeon_retador"] = float((dec == dec_r).mean())
    salida["latencia_ms"] = {"p50": float(np.percentile(lat, 50)),
                             "p95": float(np.percentile(lat, 95)),
                             "max": float(lat.max())}
    salida["serie"] = _serie(filas)
    salida["ultimas"] = [
        {k: f[k] for k in ("recibido_utc", "amount", "channel", "mcc", "p_campeon",
                           "decision", "p_retador", "decision_retador", "latencia_ms",
                           "is_fraud", "origen")}
        for f in filas[-15:][::-1]
    ]

    # -------------------------------------------------------------- deriva
    deriva = {}
    suficientes = len(filas) >= MIN_PARA_PSI
    for rol, campo in (("campeon", "score_campeon"), ("retador", "score_retador")):
        r = referencia["puntajes"][rol]
        v, actual = psi(np.array([f[campo] for f in filas]), r["cortes"], r["proporciones"])
        deriva[f"puntaje_{rol}"] = {"psi": v if suficientes else None,
                                    "nivel": _nivel_psi(v if suficientes else None),
                                    "actual": actual, "esperado": r["proporciones"]}
    r = referencia["features"]["amount"]
    v, actual = psi(np.array([f["amount"] for f in filas]), r["cortes"], r["proporciones"])
    deriva["monto"] = {"psi": v if suficientes else None, "nivel": _nivel_psi(v if suficientes else None),
                       "actual": actual, "esperado": r["proporciones"]}
    v = psi_categorico([f["channel"] for f in filas], referencia["features"]["channel"])
    deriva["canal"] = {"psi": v if suficientes else None, "nivel": _nivel_psi(v if suficientes else None)}
    salida["deriva"] = deriva

    # -------------------------------------------------------------- desempeño
    etiq = [f for f in filas if f["is_fraud"] is not None]
    salida["n_etiquetadas"] = len(etiq)
    if etiq:
        y = np.array([f["is_fraud"] for f in etiq], dtype=int)
        salida["desempeno"] = {
            "campeon": _desempeno(y, np.array([f["p_campeon"] for f in etiq]),
                                  np.array([f["decision"] for f in etiq]), costos),
            "retador": _desempeno(y, np.array([f["p_retador"] for f in etiq]),
                                  np.array([f["decision_retador"] for f in etiq]), costos),
            "costo_sin_modelo_Q": float(y.sum() * costos["FN"]),
        }

    # -------------------------------------------------------------- alertas
    if not suficientes:
        salida["alertas"].append({"nivel": "info", "mensaje":
                                  f"La deriva se calcula a partir de {MIN_PARA_PSI} predicciones "
                                  f"(hay {len(filas)})."})
    for nombre, info in deriva.items():
        if info["nivel"] in ("vigilar", "alerta"):
            salida["alertas"].append({"nivel": info["nivel"], "mensaje":
                                      f"Deriva en {nombre.replace('_', ' ')}: PSI = {info['psi']:.3f}"})
    if salida["latencia_ms"]["p95"] > LATENCIA_OBJETIVO_MS:
        salida["alertas"].append({"nivel": "vigilar", "mensaje":
                                  f"Latencia p95 = {salida['latencia_ms']['p95']:.0f} ms "
                                  f"(objetivo < {LATENCIA_OBJETIVO_MS:.0f} ms)"})
    tasa_bloqueo_test = meta["test"][roles["campeon"]["familia"]]["tasa_bloqueo"]
    if suficientes and salida["decisiones"]["block"] > 3 * max(tasa_bloqueo_test, 0.001):
        salida["alertas"].append({"nivel": "alerta", "mensaje":
                                  f"Tasa de bloqueo {salida['decisiones']['block']:.1%} "
                                  f"> 3× la esperada ({tasa_bloqueo_test:.1%})"})
    if "desempeno" in salida:
        c, r = salida["desempeno"]["campeon"], salida["desempeno"]["retador"]
        if c["n_fraude"] >= 20 and r["costo_Q"] < c["costo_Q"]:
            salida["alertas"].append({"nivel": "vigilar", "mensaje":
                                      f"El retador ({salida['retador']}) cuesta menos que el campeón "
                                      f"sobre tráfico etiquetado: Q{r['costo_Q']:,.0f} vs "
                                      f"Q{c['costo_Q']:,.0f}. Revisar en comité."})
    return salida
