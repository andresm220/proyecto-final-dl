"""Figuras del informe. Cada una guarda un PNG a 150 dpi."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from . import config as cfg


def _guardar(fig, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def curvas_pr(resultados: dict[str, tuple], ruta) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for nombre, (y, p) in resultados.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=nombre, lw=2)
    ax.set_xlabel("Exhaustividad (recall)")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas precision-exhaustividad (validacion)")
    ax.legend()
    ax.grid(alpha=0.3)
    _guardar(fig, ruta)


def auc_vs_k(ks, aucs, auc_modelo_a: float, ruta) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, aucs, "o-", lw=2, label="Modelo B (GRU)")
    ax.axhline(auc_modelo_a, ls="--", color="gray", label="Modelo A (agregados)")
    ax.set_xlabel("K - eventos de historia")
    ax.set_ylabel("AUC-PR (validacion)")
    ax.set_title("Cuanta historia hace falta?")
    ax.legend()
    ax.grid(alpha=0.3)
    _guardar(fig, ruta)


def curva_costo(curvas: dict[str, tuple], umbrales: dict[str, float], ruta) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for nombre, (u, c) in curvas.items():
        linea, = ax.plot(u, c, lw=2, label=nombre)
        ux = umbrales[nombre]
        ax.axvline(ux, ls=":", color=linea.get_color())
        ax.annotate(f"u*={ux:.3f}", (ux, np.interp(ux, u, c)),
                    textcoords="offset points", xytext=(6, 8), color=linea.get_color())
    ax.axvline(cfg.UMBRAL_TEORICO, ls="--", color="black", alpha=0.5,
               label=f"teorico {cfg.UMBRAL_TEORICO:.4f}")
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("Umbral de bloqueo")
    ax.set_ylabel("Costo esperado (Q)")
    ax.set_title("Costo vs umbral - bloquear a partir de ~4.3 %")
    ax.legend()
    ax.grid(alpha=0.3)
    _guardar(fig, ruta)
