"""AUC-PR primaria. La exactitud no se calcula ni se expone (-15 pts)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def auc_pr(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def en_umbral(y: np.ndarray, p: np.ndarray, u: float) -> dict:
    pred = (p >= u).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def desglose_por_tipo(y: np.ndarray, p: np.ndarray, subtipo: np.ndarray, u: float) -> pd.DataFrame:
    """Un mecanismo a la vez, siempre contra el total de legitimos.

    Comparar un tipo de fraude contra otros fraudes no responde ninguna
    pregunta de negocio; contra los legitimos, si.
    """
    subtipo = np.asarray(subtipo)
    legitimos = subtipo == "none"
    filas = []
    for grupo in sorted(set(subtipo.tolist()) - {"none"}):
        sel = legitimos | (subtipo == grupo)
        yg, pg = y[sel], p[sel]
        filas.append(
            {
                "grupo": grupo,
                "n": int(sel.sum()),
                "n_fraude": int(yg.sum()),
                "auc_pr": auc_pr(yg, pg) if yg.sum() else float("nan"),
                "recall": en_umbral(yg, pg, u)["recall"],
            }
        )
    return pd.DataFrame(filas)


def resumen(valores) -> tuple[float, float]:
    """Media y desviacion muestral sobre las semillas."""
    a = np.asarray(list(valores), dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0
