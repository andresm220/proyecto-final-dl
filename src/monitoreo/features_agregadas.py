"""Agregados causales del Modelo A.

Todas las features son invariantes al orden por construccion (medias,
conteos, cardinalidades de conjunto). Todos los agregados de contexto usan
`closed='left'`: `amt` es la unica feature que describe la transaccion que
se esta puntuando.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

_VENTANAS = {"1h": pd.Timedelta("1h"), "24h": pd.Timedelta("24h"), "7d": pd.Timedelta("7d")}


def _rolling_causal(g: pd.DataFrame, ventana: str, columna: str, op: str) -> pd.Series:
    s = g.set_index("ts")[columna]
    return s.rolling(ventana, closed="left").agg(op)


def _n_distintos_causal(ts: np.ndarray, comercio: np.ndarray, ventana: pd.Timedelta) -> np.ndarray:
    """Conteo de comercios distintos en (t-ventana, t), excluyendo t.

    Dos punteros con un multiconjunto incremental: exacto y O(n).
    pandas no ofrece rolling.nunique.
    """
    n = len(ts)
    out = np.zeros(n, dtype=np.int32)
    conteo: dict[int, int] = {}
    izq = 0
    for der in range(n):
        limite = ts[der] - ventana.to_timedelta64()
        while izq < der and ts[izq] < limite:
            c = int(comercio[izq])
            conteo[c] -= 1
            if conteo[c] == 0:
                del conteo[c]
            izq += 1
        out[der] = len(conteo)
        c = int(comercio[der])
        conteo[c] = conteo.get(c, 0) + 1
    return out


def construir(df: pd.DataFrame) -> pd.DataFrame:
    """X_A. Mismo numero de filas y mismo orden que `df`."""
    partes = []
    for _, g in df.groupby("card_id", sort=False):
        g = g.sort_values("ts", kind="mergesort")
        x = pd.DataFrame(index=g.index)
        x["amt"] = g["amount"].to_numpy()
        x["amt_mean_24h"] = _rolling_causal(g, "24h", "amount", "mean").to_numpy()
        x["amt_std_24h"] = _rolling_causal(g, "24h", "amount", "std").to_numpy()
        x["amt_max_24h"] = _rolling_causal(g, "24h", "amount", "max").to_numpy()
        x["n_tx_1h"] = _rolling_causal(g, "1h", "amount", "count").to_numpy()
        x["n_tx_24h"] = _rolling_causal(g, "24h", "amount", "count").to_numpy()
        media_7d = _rolling_causal(g, "7d", "amount", "mean").to_numpy()
        x["n_merchants_24h"] = _n_distintos_causal(
            g["ts"].to_numpy(), g["merchant_id"].to_numpy(), _VENTANAS["24h"]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            x["amt_ratio_to_mean_7d"] = np.where(
                np.isfinite(media_7d) & (media_7d > 0), g["amount"].to_numpy() / media_7d, 1.0
            )
        partes.append(x)

    X = pd.concat(partes).reindex(df.index)
    X = X.fillna(0.0)

    hora = df["ts"].dt.hour.to_numpy() + df["ts"].dt.minute.to_numpy() / 60.0
    X["hour_sin"] = np.sin(2 * np.pi * hora / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hora / 24.0)
    X["is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(float).to_numpy()

    for canal in cfg.CANALES:
        X[f"channel_{canal}"] = (df["channel"] == canal).astype(float).to_numpy()
    for mcc in cfg.MCCS:
        X[f"mcc_{mcc}"] = (df["mcc"] == mcc).astype(float).to_numpy()

    verificar_sin_orden(X)
    return X.astype(np.float32)


def verificar_sin_orden(X: pd.DataFrame) -> None:
    """Una feature de orden en A contaminaria la comparacion con B."""
    for col in X.columns:
        bajo = col.lower()
        for patron in cfg.PATRONES_PROHIBIDOS_EN_A:
            assert patron not in bajo, f"'{col}' parece codificar orden; A debe ser ciego al orden"
    for col in cfg.COLUMNAS_ANALISIS:
        assert col not in X.columns, f"'{col}' es solo para analisis, nunca feature"
