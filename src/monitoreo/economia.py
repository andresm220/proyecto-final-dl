"""Umbral por costo, no por F1 ni por percentil.

Dejar pasar un fraude cuesta 23 veces mas que molestar a un cliente
legitimo, asi que el umbral optimo esta lejisimos de 0.5.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg

DIAS_DEL_MES = 30.0


def costo(y: np.ndarray, p: np.ndarray, u: float) -> float:
    pred = p >= u
    fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return fn * cfg.COSTO_FN + fp * cfg.COSTO_FP


def curva(y: np.ndarray, p: np.ndarray, n_pasos: int = 1000):
    """Barrido de u en [0,1]. Vectorizado sobre el orden de los puntajes."""
    umbrales = np.linspace(0.0, 1.0, n_pasos)
    orden = np.argsort(p)
    y_ord = y[orden]
    p_ord = p[orden]
    # nro de positivos y negativos por debajo de cada umbral
    corte = np.searchsorted(p_ord, umbrales, side="left")
    pos_acum = np.concatenate([[0], np.cumsum(y_ord == 1)])
    neg_acum = np.concatenate([[0], np.cumsum(y_ord == 0)])
    fn = pos_acum[corte]                       # positivos que quedan por debajo
    fp = neg_acum[-1] - neg_acum[corte]        # negativos que quedan por encima
    return umbrales, fn * cfg.COSTO_FN + fp * cfg.COSTO_FP


def umbral_optimo(y: np.ndarray, p: np.ndarray, n_pasos: int = 1000) -> tuple[float, float]:
    """u* que minimiza el costo. Se elige SOBRE VALIDACION y se congela."""
    umbrales, costos = curva(y, p, n_pasos)
    i = int(np.argmin(costos))
    return float(umbrales[i]), float(costos[i])


def ahorro_mensual(costo_a: float, costo_b: float, dias_test: float) -> float:
    """Extrapolacion explicita del periodo de test a un mes de 30 dias."""
    return (costo_a - costo_b) * DIAS_DEL_MES / dias_test
