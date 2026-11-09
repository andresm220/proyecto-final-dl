"""Semillas y captura de versiones para el README."""
from __future__ import annotations

import os
import platform
import random
import sys


def fijar_semillas(seed: int) -> None:
    """Fija numpy, random y el backend de Keras si esta disponible.

    `keras.utils.set_random_seed` cubre python, numpy y el backend activo
    (torch en esta maquina), asi que no hace falta sembrar cada uno aparte.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import keras
    except ImportError:
        return
    keras.utils.set_random_seed(seed)


def versiones() -> dict[str, str]:
    """Versiones exactas para el README y artefactos/config.json."""
    import importlib.metadata as md

    v = {
        "python": sys.version.split()[0],
        "sistema": f"{platform.system()} {platform.release()}",
    }
    for paquete in ("numpy", "pandas", "scikit-learn", "keras", "torch", "lightgbm"):
        try:
            v[paquete] = md.version(paquete)
        except md.PackageNotFoundError:
            v[paquete] = "no instalado"
    try:
        import keras

        v["keras_backend"] = keras.backend.backend()
    except ImportError:
        v["keras_backend"] = "no instalado"
    return v
