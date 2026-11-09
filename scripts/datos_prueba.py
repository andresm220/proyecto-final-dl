"""Acceso al periodo de PRUEBA para armar solicitudes reales a la API.

Regenera el mismo dataset del Proyecto 1 (misma semilla) y, para cualquier evento,
arma la solicitud que mandaría el sistema del banco: la historia de la tarjeta de los
7 días previos (y al menos las 3 últimas) más la transacción a puntuar.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from monitoreo import config as cfg  # noqa: E402
from monitoreo import generador as gen  # noqa: E402
from monitoreo import particion as part  # noqa: E402

CAMPOS = ("ts", "amount", "merchant_id", "mcc", "channel", "country")


def cargar() -> tuple[pd.DataFrame, np.ndarray]:
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=cfg.n_tarjetas())
    es_test = (part.asignar_split(df) == "test").to_numpy()
    return df, es_test


def _tx(fila, factor_monto: float = 1.0) -> dict:
    d = {c: fila[c] for c in CAMPOS}
    d["ts"] = pd.Timestamp(d["ts"]).isoformat() + "Z"
    d["amount"] = round(float(d["amount"]) * factor_monto, 2)
    d["merchant_id"] = int(d["merchant_id"])
    return d


def solicitud(df: pd.DataFrame, i: int, factor_monto: float = 1.0, minimo: int = 3) -> dict:
    """Solicitud para el evento en la posición i (df ordenado por card_id, ts)."""
    fila = df.iloc[i]
    tarjeta = df[df["card_id"] == fila["card_id"]]
    previas = tarjeta[tarjeta.index < df.index[i]]
    recientes = (previas["ts"] >= fila["ts"] - pd.Timedelta("7d")).to_numpy()
    if len(previas):
        recientes[-minimo:] = True
    previas = previas[recientes]
    return {
        "card_id": int(fila["card_id"]),
        "historial": [_tx(r, factor_monto) for _, r in previas.iterrows()],
        "transaccion": _tx(fila, factor_monto),
    }
