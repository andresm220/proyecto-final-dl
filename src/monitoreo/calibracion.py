"""Calibracion isotonica ajustada en VALIDACION.

Sin esto, el analisis economico compara peras con manzanas entre A y B:
los puntajes de una red con class_weight no son probabilidades.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def ajustar(p_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)


def aplicar(cal: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return np.clip(cal.predict(p), 0.0, 1.0)
