"""Ventanas deslizantes representadas como INDICES, no como tensores.

Guardar indices en vez de floats hace que la prueba de permutacion preserve
el contenido por construccion: barajar es permutar enteros dentro de una
fila, y es imposible que altere QUE eventos hay en la ventana.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def _verificar_orden(df: pd.DataFrame) -> None:
    """cumcount depende del orden por (card_id, ts). Sin esta guarda, un frame
    desordenado produce indices silenciosamente incorrectos."""
    orden = df[["card_id", "ts"]]
    assert orden.equals(orden.sort_values(["card_id", "ts"], kind="mergesort")), (
        "df debe venir ordenado por (card_id, ts); usar generador.generar o "
        "df.sort_values(['card_id','ts']).reset_index(drop=True)"
    )


def construir(df: pd.DataFrame, K: int = cfg.K) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (win_idx, mask).

    `df` debe venir ordenado por (card_id, ts). Los indices son POSICIONALES
    respecto de `df`, de 0 a len(df)-1.
    """
    _verificar_orden(df)
    n = len(df)
    pos_global = np.arange(n, dtype=np.int32)
    pos_en_tarjeta = df.reset_index(drop=True).groupby("card_id").cumcount().to_numpy()

    desplazamientos = np.arange(K - 1, -1, -1, dtype=np.int32)      # K-1 ... 0
    candidatos = pos_global[:, None] - desplazamientos[None, :]
    disponible = pos_en_tarjeta[:, None] >= desplazamientos[None, :]

    win = np.where(disponible, candidatos, pos_global[:, None]).astype(np.int32)
    return win, disponible


MODOS = ("full", "history")


def permutar(
    win_idx: np.ndarray, mask: np.ndarray, modo: str, rng: np.random.Generator
) -> np.ndarray:
    """Baraja el orden de la ventana sin alterar su contenido.

    - "full":    baraja las K posiciones validas, evento objetivo incluido.
    - "history": baraja las K-1 previas y deja el objetivo en la ultima
                 posicion. Aisla el aporte del orden de la HISTORIA.

    Solo se permutan posiciones validas: si el padding se mezclara al
    centro, la mascara dejaria de describir la secuencia.
    """
    if modo not in MODOS:
        raise ValueError(f"modo debe ser uno de {MODOS}, no {modo!r}")

    perm = win_idx.copy()
    ultima = win_idx.shape[1] - 1

    for i in range(win_idx.shape[0]):
        posiciones = np.flatnonzero(mask[i])
        if modo == "history":
            posiciones = posiciones[posiciones != ultima]
        if posiciones.size < 2:
            continue
        perm[i, posiciones] = win_idx[i, rng.permutation(posiciones)]

    return perm
