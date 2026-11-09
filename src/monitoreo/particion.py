"""Corte temporal global. Penalizacion de -20 pts por particion aleatoria."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def asignar_split(df: pd.DataFrame) -> pd.Series:
    """Percentiles de `ts` GLOBAL, no por tarjeta.

    Se usa el rango ordinal y no `quantile` sobre las fechas para que los
    empates de timestamp no desbalanceen los cortes.
    """
    orden = df["ts"].rank(method="first")
    n = len(df)
    corte_tr = cfg.PCT_TRAIN * n
    corte_val = cfg.PCT_VAL * n
    s = pd.Series("test", index=df.index, dtype=object)
    s[orden <= corte_val] = "val"
    s[orden <= corte_tr] = "train"
    return s


def tabla(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Tabla 1 del informe: tamano, fechas de corte y tasa de fraude."""
    filas = []
    for nombre in ("train", "val", "test"):
        g = df[split == nombre]
        filas.append(
            {
                "split": nombre,
                "n": len(g),
                "fecha_min": g["ts"].min(),
                "fecha_max": g["ts"].max(),
                "n_fraude": int(g["is_fraud"].sum()),
                "tasa_fraude": float(g["is_fraud"].mean()),
            }
        )
    return pd.DataFrame(filas)
