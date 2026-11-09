"""Matriz de eventos para el Modelo B.

`delta_t`, `same_merchant_as_prev` y `amount_ratio_to_prev` se calculan
sobre el flujo ORIGINAL, antes de ventanear: al barajar la ventana cada
evento se lleva consigo su delta_t. Ver seccion 8.2 del spec -- por eso
existe la ablacion de delta_t como tercera comprobacion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as cfg

NOMBRES_NUM = (
    "log_amount",
    "log_delta_t",
    "es_primera",
    "hour_sin",
    "hour_cos",
    "same_merchant_as_prev",
    "amount_ratio_to_prev",
)


def _verificar_orden(df: pd.DataFrame) -> None:
    """Delta_t, same_merchant_as_prev y amount_ratio_to_prev se calculan con
    groupby+shift, asi que dependen de que el frame venga ordenado por
    (card_id, ts). Sin esta guarda, un frame desordenado produce features
    silenciosamente incorrectas en vez de fallar."""
    orden = df[["card_id", "ts"]]
    assert orden.equals(orden.sort_values(["card_id", "ts"], kind="mergesort")), (
        "df debe venir ordenado por (card_id, ts); usar generador.generar o "
        "df.sort_values(['card_id','ts']).reset_index(drop=True)"
    )


def construir_vocabularios(df: pd.DataFrame, es_train: np.ndarray) -> dict[str, dict]:
    """Indices >= 2. El 0 es PAD y el 1 es UNK; deben ser distintos."""
    tr = df[es_train]
    vocab = {}
    for nombre, columna in (("mcc", "mcc"), ("channel", "channel"), ("merchant", "merchant_id")):
        categorias = sorted(tr[columna].unique().tolist())
        vocab[nombre] = {c: i + 2 for i, c in enumerate(categorias)}
    return vocab


def _codificar(serie: pd.Series, tabla: dict) -> np.ndarray:
    return serie.map(tabla).fillna(cfg.UNK).to_numpy(dtype=np.int32)


def construir(
    df: pd.DataFrame,
    vocab: dict[str, dict],
    es_train: np.ndarray,
    scaler: StandardScaler | None = None,
    usar_delta_t: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Devuelve (E_num, E_cat, scaler). El scaler se ajusta SOLO con train."""
    _verificar_orden(df)
    g = df.groupby("card_id", sort=False)
    delta = g["ts"].diff().dt.total_seconds()
    es_primera = delta.isna().to_numpy().astype(np.float32)
    delta = delta.fillna(0.0).clip(lower=0.0).to_numpy()

    prev_merchant = g["merchant_id"].shift()
    prev_amount = g["amount"].shift()
    hora = df["ts"].dt.hour.to_numpy() + df["ts"].dt.minute.to_numpy() / 60.0

    columnas = {
        "log_amount": np.log1p(df["amount"].to_numpy()),
        "log_delta_t": np.log1p(delta),
        "es_primera": es_primera,
        "hour_sin": np.sin(2 * np.pi * hora / 24.0),
        "hour_cos": np.cos(2 * np.pi * hora / 24.0),
        "same_merchant_as_prev": (df["merchant_id"] == prev_merchant).astype(float).to_numpy(),
        "amount_ratio_to_prev": (
            df["amount"] / prev_amount.replace(0.0, np.nan)
        ).fillna(1.0).clip(upper=100.0).to_numpy(),
    }
    nombres = [n for n in NOMBRES_NUM if usar_delta_t or n != "log_delta_t"]
    E_num = np.column_stack([columnas[n] for n in nombres]).astype(np.float32)

    if scaler is None:
        scaler = StandardScaler().fit(E_num[es_train])
    E_num = scaler.transform(E_num).astype(np.float32)

    E_cat = np.column_stack(
        [
            _codificar(df["mcc"], vocab["mcc"]),
            _codificar(df["channel"], vocab["channel"]),
            _codificar(df["merchant_id"], vocab["merchant"]),
        ]
    ).astype(np.int32)

    return E_num, E_cat, scaler


def cardinalidades(vocab: dict[str, dict]) -> dict[str, int]:
    """Tamano de cada tabla de embedding, contando PAD y UNK."""
    return {k: max(v.values()) + 1 for k, v in vocab.items()}
