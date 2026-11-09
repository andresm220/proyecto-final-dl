"""Entrenamiento reproducible del paquete de producción.

Uso:
    python -m entrenamiento.entrenar            # corrida completa (~1 h en CPU)
    MONITOREO_DEV=1 python -m entrenamiento.entrenar   # 400 tarjetas, para probar

Qué hace, en orden — y por qué en ese orden:
  1. Registra el SHA-256 de docs/HIPOTESIS.md y la hora de inicio. Las hipótesis y
     la regla de decisión se congelaron antes de esta corrida.
  2. Regenera los datos del Proyecto 1 con la misma semilla (mismo dataset exacto).
  3. Entrena A (LightGBM), B_K3 (GRU) y C_K3 (híbrido) con las semillas (7, 13, 29),
     mirando solo train y validación.
  4. Evalúa H1 y H2 y elige campeón y retador con la regla congelada.
  5. Calibra y fija u* en validación; define la banda de revisión.
  6. Prueba de permutación sobre el retador/campeón secuencial (falsificación).
  7. Evalúa test UNA vez, ya con todo congelado, y solo lo reporta.
  8. Guarda modelos, preprocesadores, umbrales, referencia de monitoreo y metadata.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np  # noqa: E402

from monitoreo import calibracion as cal  # noqa: E402
from monitoreo import config as cfg  # noqa: E402
from monitoreo import economia as eco  # noqa: E402
from monitoreo import experimentos as ex  # noqa: E402
from monitoreo import metricas as met  # noqa: E402
from monitoreo import reproducibilidad as rep  # noqa: E402

K_PROD = 3
SALIDA = RAIZ / "artefactos" / "produccion"
CACHE = RAIZ / "artefactos" / "cache_entrenamiento"
HIPOTESIS = RAIZ / "docs" / "HIPOTESIS.md"

# Cifras de partida del Proyecto 1 (validación, media de 3 semillas, K = 20).
P1_B_K20_MEDIA = 0.7137847506722514
P1_A_MEDIA = 0.8328582732704707

# Capacidad supuesta del equipo de analistas: la banda de revisión manual no
# debe mandar a la cola más del 1 % del tráfico. Supuesto de negocio, ver README §3.
CAPACIDAD_REVISION = 0.01

# Ensamble (adenda H3 de docs/HIPOTESIS.md): peso fijo, no se ajusta en validación.
PESO_A = 0.5


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _banda_revision(p_cal_val: np.ndarray, u_estrella: float) -> float:
    """Límite inferior de la banda 'revisar'.

    Se toman, por debajo de u*, los puntajes más altos hasta llenar la capacidad
    de revisión. Así la cola de analistas tiene un tamaño acotado y conocido.
    """
    debajo = np.sort(p_cal_val[p_cal_val < u_estrella])[::-1]
    n_cupo = int(CAPACIDAD_REVISION * len(p_cal_val))
    if n_cupo == 0 or len(debajo) == 0:
        return u_estrella
    return float(debajo[min(n_cupo, len(debajo)) - 1])


def _referencia_monitoreo(d: dict, p_raw: dict[str, np.ndarray]) -> dict:
    """Distribuciones de validación contra las que el servicio mide deriva (PSI)."""
    va = d["va"]
    df_va = d["df"][va]
    ref = {"puntajes": {}, "features": {}}
    for nombre, p in p_raw.items():
        cortes = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        conteo, _ = np.histogram(p, bins=cortes)
        ref["puntajes"][nombre] = {"cortes": cortes.tolist(),
                                   "proporciones": (conteo / conteo.sum()).tolist()}
    monto = df_va["amount"].to_numpy()
    cortes = np.unique(np.quantile(monto, np.linspace(0, 1, 11)))
    conteo, _ = np.histogram(monto, bins=cortes)
    ref["features"]["amount"] = {"cortes": cortes.tolist(),
                                 "proporciones": (conteo / conteo.sum()).tolist()}
    ref["features"]["channel"] = df_va["channel"].value_counts(normalize=True).to_dict()
    ref["tasa_fraude_val"] = float(d["y"][va].mean())
    return ref


def main() -> None:
    t0 = time.time()
    inicio = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    sha_hip = _sha256(HIPOTESIS)
    _log(f"Hipótesis congeladas: sha256={sha_hip[:12]}…  inicio={inicio}")
    _log(f"Modo dev: {cfg.dev_mode()}  tarjetas={cfg.n_tarjetas()}  K={K_PROD}")

    # ------------------------------------------------------------ datos
    d = ex.preparar(K=K_PROD)
    ex.verificar_contrato(d)
    y, va, te = d["y"], d["va"], d["te"]
    _log(f"Datos: {len(d['df']):,} eventos, train/val/test = "
         f"{int(d['tr'].sum()):,}/{int(va.sum()):,}/{int(te.sum()):,}")

    sufijo = "_dev" if cfg.dev_mode() else ""
    cache = CACHE / f"K{K_PROD}{sufijo}"

    # ------------------------------------------------------------ A
    ra = ex.correr_a(d)
    _log(f"A (LightGBM) AUC-PR val por semilla: {np.round(ra['gbm'], 4).tolist()}")

    # ------------------------------------------------------------ B y C
    res = {"B_K3": [], "C_K3": []}
    modelos = {"B_K3": [], "C_K3": []}
    pval = {"B_K3": [], "C_K3": []}
    for s in cfg.SEEDS_MODELO:
        for fam, hib in (("B_K3", False), ("C_K3", True)):
            r = ex.correr_b_cacheado(d, s, ruta=cache / f"{fam}_semilla{s}.keras",
                                     hibrido=hib, verbose=0)
            res[fam].append(r["auc_pr"])
            modelos[fam].append(r["modelo"])
            pval[fam].append(r["p_val"])
            origen = "cache" if r["desde_cache"] else f"{r['segundos']:.0f} s"
            _log(f"{fam} semilla {s}: AUC-PR val = {r['auc_pr']:.4f} ({origen})")

    medias = {"A": float(np.mean(ra["gbm"])),
              "B_K3": float(np.mean(res["B_K3"])),
              "C_K3": float(np.mean(res["C_K3"]))}
    desv = {"A": float(np.std(ra["gbm"])),
            "B_K3": float(np.std(res["B_K3"])),
            "C_K3": float(np.std(res["C_K3"]))}

    # ------------------------------------------------------------ hipótesis
    h1_mejora = medias["B_K3"] - P1_B_K20_MEDIA
    h1 = {"mejora_media": h1_mejora, "peor_semilla": float(min(res["B_K3"])),
          "exito": bool(h1_mejora >= 0.03 and min(res["B_K3"]) > P1_B_K20_MEDIA)}
    h2 = {"diferencia_vs_A": medias["C_K3"] - medias["A"],
          "exito": bool(medias["C_K3"] >= medias["A"] - 0.005)}
    _log(f"H1 (B_K3 > B_K20): mejora {h1_mejora:+.4f} → {'ÉXITO' if h1['exito'] else 'NO'}")
    _log(f"H2 (C_K3 ≥ A):     dif {h2['diferencia_vs_A']:+.4f} → {'ÉXITO' if h2['exito'] else 'NO'}")

    # ------------------------------------------------------------ H3: ensamble
    # Pares con la misma semilla. Peso fijo 0.5 sobre puntajes CRUDOS (docs/HIPOTESIS.md,
    # adenda): nada del ensamble se ajusta mirando validación, salvo su calibración y u*.
    def costo_en_ustar(p_val):
        p_cal = cal.aplicar(cal.ajustar(p_val, y[va]), p_val)
        return eco.umbral_optimo(y[va], p_cal)[1]

    pval_e, res_e, costo_e, costo_a = [], [], [], []
    for i, s in enumerate(cfg.SEEDS_MODELO):
        pe = PESO_A * ra["p_val"][s] + (1 - PESO_A) * pval["C_K3"][i]
        pval_e.append(pe)
        res_e.append(met.auc_pr(y[va], pe))
        costo_e.append(costo_en_ustar(pe))
        costo_a.append(costo_en_ustar(ra["p_val"][s]))
        _log(f"E (A{s} + C_K3 {s}): AUC-PR val = {res_e[-1]:.4f}  "
             f"costo val E Q{costo_e[-1]:,.0f} vs A Q{costo_a[-1]:,.0f}")
    medias["E"], desv["E"] = float(np.mean(res_e)), float(np.std(res_e))
    h3 = {"diferencia_vs_A": medias["E"] - medias["A"],
          "costo_val_medio_E": float(np.mean(costo_e)),
          "costo_val_medio_A": float(np.mean(costo_a)),
          "exito": bool(medias["E"] >= medias["A"]
                        and np.mean(costo_e) <= np.mean(costo_a))}
    _log(f"H3 (E no peor que A): dif {h3['diferencia_vs_A']:+.4f}, costo medio "
         f"Q{h3['costo_val_medio_E']:,.0f} vs Q{h3['costo_val_medio_A']:,.0f} "
         f"→ {'ÉXITO' if h3['exito'] else 'NO'}")

    # ------------------------------------------------------------ regla de decisión (adenda)
    campeon, retador = ("E", "A") if h3["exito"] else ("A", "E")
    _log(f"Campeón: {campeon}   Retador (sombra): {retador}")

    def mejor(fam):
        """Componentes, puntajes de validación y semilla del mejor de cada familia."""
        if fam == "A":
            i = int(np.argmax(ra["gbm"]))
            s = cfg.SEEDS_MODELO[i]
            return {"A": ra["modelos_gbm"][i]}, ra["p_val"][s], s
        if fam == "E":
            i = int(np.argmax(res_e))
            return ({"A": ra["modelos_gbm"][i], "C_K3": modelos["C_K3"][i]},
                    pval_e[i], cfg.SEEDS_MODELO[i])
        i = int(np.argmax(res[fam]))
        return {fam: modelos[fam][i]}, pval[fam][i], cfg.SEEDS_MODELO[i]

    def crudo(comp: dict, sel: np.ndarray) -> np.ndarray:
        """Puntaje crudo de un modelo (simple o ensamble) sobre un subconjunto."""
        partes = {}
        if "A" in comp:
            partes["A"] = comp["A"].predict_proba(d["X_A"].to_numpy()[sel])[:, 1]
        for fam in ("B_K3", "C_K3"):
            if fam in comp:
                partes[fam] = ex.predecir_split(comp[fam], d, sel, hibrido=fam.startswith("C"))
        if len(partes) == 1:
            return next(iter(partes.values()))
        return PESO_A * partes["A"] + (1 - PESO_A) * partes["C_K3"]

    servidos = {}
    for rol, fam in (("campeon", campeon), ("retador", retador)):
        comp, p_val, s = mejor(fam)
        calib = cal.ajustar(p_val, y[va])
        p_cal = cal.aplicar(calib, p_val)
        u, costo_val = eco.umbral_optimo(y[va], p_cal)
        servidos[rol] = {"familia": fam, "componentes": comp, "semilla": s, "p_val": p_val,
                         "calibrador": calib, "u_estrella": u, "costo_val": costo_val,
                         "u_revision": _banda_revision(p_cal, u),
                         "auc_pr_val": met.auc_pr(y[va], p_val)}
        _log(f"{rol}={fam} semilla {s}: u*={u:.4f}  revisión desde "
             f"{servidos[rol]['u_revision']:.4f}  costo val Q{costo_val:,.0f}")

    # ------------------------------------------------------------ permutación
    # Sobre el C_K3 que integra el ensamble: es la parte que lee el orden.
    _, _, s_e = mejor("E")
    c_perm = modelos["C_K3"][cfg.SEEDS_MODELO.index(s_e)]
    tabla_perm = ex.tabla_permutacion(c_perm, d, ra["p_val"][s_e], hibrido=True)
    _log("Permutación del C_K3 del ensamble (validación):\n" + tabla_perm.to_string(index=False))

    # ------------------------------------------------------------ test, una vez
    fecha_test = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    dias_test = (d["df"]["ts"][te].max() - d["df"]["ts"][te].min()).total_seconds() / 86400
    resultados_test = {}
    for rol, s in servidos.items():
        p_te = crudo(s["componentes"], te)
        p_te_cal = cal.aplicar(s["calibrador"], p_te)
        u = s["u_estrella"]
        resultados_test[s["familia"]] = {
            "auc_pr": met.auc_pr(y[te], p_te),
            "costo_Q": eco.costo(y[te], p_te_cal, u),
            "en_umbral": met.en_umbral(y[te], p_te_cal, u),
            "por_tipo": met.desglose_por_tipo(y[te], p_te_cal, d["subtipo"][te], u)
                           .to_dict(orient="records"),
            "tasa_bloqueo": float((p_te_cal >= u).mean()),
            "tasa_revision": float(((p_te_cal >= s["u_revision"]) & (p_te_cal < u)).mean()),
        }
    costo_sin_modelo = float(y[te].sum() * cfg.COSTO_FN)
    for fam, r in resultados_test.items():
        r["ahorro_mensual_vs_sin_modelo_Q"] = eco.ahorro_mensual(
            costo_sin_modelo, r["costo_Q"], dias_test)
    _log(f"TEST (fecha {fecha_test}): " + json.dumps(
        {k: {"auc_pr": round(v["auc_pr"], 4), "costo_Q": v["costo_Q"]}
         for k, v in resultados_test.items()}))

    # ------------------------------------------------------------ guardar
    if SALIDA.exists():
        shutil.rmtree(SALIDA)
    SALIDA.mkdir(parents=True)
    for rol, s in servidos.items():
        s["archivos"] = {}
        for fam, m in s["componentes"].items():
            if fam == "A":
                nombre = f"{rol}_{s['familia']}_A.pkl" if s["familia"] != "A" else f"{rol}_A.pkl"
                with open(SALIDA / nombre, "wb") as f:
                    pickle.dump(m, f)
            else:
                nombre = (f"{rol}_{s['familia']}_{fam}.keras" if s["familia"] != fam
                          else f"{rol}_{fam}.keras")
                m.save(SALIDA / nombre)
            s["archivos"][fam] = nombre
    with open(SALIDA / "preprocesadores.pkl", "wb") as f:
        pickle.dump({"scaler_eventos": d["scaler"], "scaler_agregados": d["esc_agg"],
                     "calibradores": {rol: s["calibrador"] for rol, s in servidos.items()}}, f)
    (SALIDA / "vocabularios.json").write_text(
        json.dumps({k: {str(c): i for c, i in v.items()} for k, v in d["vocab"].items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (SALIDA / "referencia_monitoreo.json").write_text(json.dumps(
        _referencia_monitoreo(d, {rol: s["p_val"] for rol, s in servidos.items()}),
        indent=2), encoding="utf-8")

    version = f"{dt.datetime.now(dt.timezone.utc):%Y%m%d}-{campeon}-vs-{retador}"
    meta = {
        "version_modelo": version,
        "hipotesis_sha256": sha_hip,
        "inicio_entrenamiento": inicio,
        "fecha_evaluacion_test": fecha_test,
        "dev_mode": cfg.dev_mode(),
        "K": K_PROD,
        "costos": {"FN": cfg.COSTO_FN, "FP": cfg.COSTO_FP},
        "capacidad_revision": CAPACIDAD_REVISION,
        "validacion": {
            "auc_pr_por_semilla": {"A": ra["gbm"], "B_K3": res["B_K3"], "C_K3": res["C_K3"],
                                   "E": res_e},
            "media": medias, "desviacion": desv,
            "referencia_P1": {"A": P1_A_MEDIA, "B_K20": P1_B_K20_MEDIA},
        },
        "hipotesis": {"H1": h1, "H2": h2, "H3": h3},
        "ensamble": {"peso_A": PESO_A, "peso_C_K3": 1 - PESO_A,
                     "definicion": "PESO_A * crudo(A) + (1 - PESO_A) * crudo(C_K3)"},
        "roles": {rol: {k: s[k] for k in ("familia", "semilla", "archivos", "u_estrella",
                                          "u_revision", "costo_val", "auc_pr_val")}
                  for rol, s in servidos.items()},
        "permutacion_val": {"modelo": f"C_K3 semilla {s_e} (componente del ensamble)",
                            "tabla": tabla_perm.to_dict(orient="records")},
        "test": resultados_test,
        "test_info": {"dias": dias_test, "n": int(te.sum()), "n_fraude": int(y[te].sum()),
                      "costo_sin_modelo_Q": costo_sin_modelo},
        "versiones": rep.versiones(),
        "duracion_min": (time.time() - t0) / 60,
    }
    (SALIDA / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    _log(f"Listo en {meta['duracion_min']:.1f} min → {SALIDA}")


if __name__ == "__main__":
    main()
