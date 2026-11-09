"""Parametros congelados del Proyecto 1. Nada aqui se decide mirando test."""
from __future__ import annotations

import os
from pathlib import Path

SEED_DATOS = 20260831
SEEDS_MODELO = (7, 13, 29)

K = 20

N_TARJETAS = 4000
N_TARJETAS_DEV = 400
TX_MIN, TX_MAX = 60, 200
TX_MEDIA = 100

TASA_FRAUDE = 0.012
MEZCLA_FRAUDE = {"f1": 0.40, "f2": 0.35, "f3": 0.25}
RAFAGAS_LEGITIMAS_POR_F1 = 3.0
PROB_F1_BRECHA_LARGA = 0.15
HORIZONTE_DIAS = 90

N_COMERCIOS = 300
MCCS = (
    "supermercado", "restaurante", "combustible", "farmacia", "ropa",
    "electronica", "hogar", "viajes", "entretenimiento", "salud",
    "educacion", "telecom", "transporte", "belleza", "ferreteria",
)
CANALES = ("POS", "online", "ATM", "recurrente")
PAISES = ("GT", "US", "MX", "ES", "CR")

PCT_TRAIN, PCT_VAL = 0.70, 0.85

COSTO_FN = 4200.0
COSTO_FP = 180.0
UMBRAL_TEORICO = COSTO_FP / COSTO_FN

DIM_EMB = {"mcc": 8, "channel": 4, "merchant": 16}
PAD, UNK = 0, 1

# 2048 y no 512: el GRU de Keras-torch en CPU va 2x mas rapido por epoca
# (46 s vs 102 s a 420k eventos) y con 1.2 % de fraude cada lote lleva ~24
# positivos en vez de ~6, lo que estabiliza el gradiente de la minoritaria.
BATCH_SIZE = 2048
UNIDADES_GRU = 64
DROPOUT = 0.3
LR = 1e-3
PACIENCIA = 5
EPOCAS_MAX = 30

RAIZ = Path(__file__).resolve().parents[2]
DIR_ARTEFACTOS = RAIZ / "artefactos"
DIR_DATOS = RAIZ / "datos"
DIR_FIGURAS = RAIZ / "informe" / "figuras"

COLUMNAS_ANALISIS = ("fraud_type", "fraud_subtype")
PATRONES_PROHIBIDOS_EN_A = ("prev", "lag", "delta", "diff", "anterior", "orden", "seq")


def dev_mode() -> bool:
    """MONITOREO_DEV=0 / "" / ausente apagan el modo rapido."""
    return os.environ.get("MONITOREO_DEV", "").strip().lower() in {"1", "true", "si", "yes"}


def n_tarjetas() -> int:
    return N_TARJETAS_DEV if dev_mode() else N_TARJETAS
