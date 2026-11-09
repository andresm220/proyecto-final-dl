"""Genera servicio/escenarios.json: casos reales del periodo de prueba para la demo.

Los casos se eligen al azar con semilla fija dentro de cada tipo, SIN mirar el
puntaje del modelo: la demo muestra lo que el modelo hace, aciertos y fallos.

    python -m scripts.generar_escenarios
"""
from __future__ import annotations

import json

import numpy as np

from scripts.datos_prueba import RAIZ, cargar, solicitud

TIPOS = [
    ("none", "Compra cotidiana", "Una transacción legítima cualquiera del periodo de prueba."),
    ("f1_golpe", "Sondeo y golpe",
     "El atacante prueba la tarjeta con montos pequeños y crecientes y luego hace una compra grande."),
    ("f2", "Ráfaga en cajero", "Varios retiros de cajero seguidos en poco tiempo."),
    ("f3", "Monto atípico aislado", "Una compra muy por encima de lo que la tarjeta suele gastar."),
]


def main() -> None:
    df, es_test = cargar()
    rng = np.random.default_rng(2026)
    salida = []
    for subtipo, titulo, descripcion in TIPOS:
        candidatos = np.flatnonzero(es_test & (df["fraud_subtype"] == subtipo).to_numpy())
        for k, i in enumerate(rng.choice(candidatos, size=2, replace=False)):
            s = solicitud(df, int(i))
            salida.append({
                "id": f"{subtipo}_{k + 1}",
                "titulo": f"{titulo} #{k + 1}",
                "descripcion": descripcion,
                "etiqueta_real": "fraude" if subtipo != "none" else "legítima",
                **s,
            })
    ruta = RAIZ / "servicio" / "escenarios.json"
    ruta.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(salida)} escenarios → {ruta}")


if __name__ == "__main__":
    main()
