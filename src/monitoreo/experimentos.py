"""Orquestacion de los experimentos del informe.

El notebook no define logica: llama a estas funciones y presenta el
resultado. Todo lo que decide algo lo hace mirando VALIDACION; la unica
funcion que toca test es la del paso final, y el notebook la invoca una
sola vez, con el umbral ya congelado.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from . import features_agregadas as fa
from . import features_evento as fe
from . import generador as gen
from . import metricas as met
from . import modelos_a as ma
from . import modelos_b as mb
from . import particion as part
from . import reproducibilidad as rep
from . import ventanas as ven


# --------------------------------------------------------------- datos


def preparar(n_tarjetas: int | None = None, K: int = cfg.K,
             usar_delta_t: bool = True) -> dict:
    """Construye el indice canonico unico que consumen A y B.

    Devuelve un dict con `df`, las mascaras de split, `X_A` (agregados),
    `X_A_esc` (los mismos escalados, para la entrada densa del hibrido),
    `E_num`/`E_cat`, `win`/`mask` y los vocabularios.
    """
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=n_tarjetas or cfg.n_tarjetas())
    split = part.asignar_split(df)
    tr = (split == "train").to_numpy()
    va = (split == "val").to_numpy()
    te = (split == "test").to_numpy()

    X_A = fa.construir(df)
    # El hibrido concatena los agregados a un estado oculto acotado; sin
    # escalar, un monto en quetzales domina al GRU. El scaler se ajusta
    # SOLO con train, igual que el de los eventos.
    esc_agg = StandardScaler().fit(X_A.to_numpy()[tr])
    X_A_esc = esc_agg.transform(X_A.to_numpy()).astype(np.float32)

    vocab = fe.construir_vocabularios(df, tr)
    E_num, E_cat, scaler = fe.construir(df, vocab, tr, usar_delta_t=usar_delta_t)
    win, mask = ven.construir(df, K)

    return dict(
        df=df, split=split, tr=tr, va=va, te=te,
        y=df["is_fraud"].to_numpy(),
        subtipo=df["fraud_subtype"].to_numpy(),
        X_A=X_A, X_A_esc=X_A_esc, esc_agg=esc_agg,
        E_num=E_num, E_cat=E_cat, scaler=scaler, vocab=vocab,
        win=win, mask=mask, K=K,
    )


def verificar_contrato(d: dict) -> None:
    """Los tres invariantes del contrato de datos, a la vista del comite."""
    n = len(d["df"])
    assert len(d["X_A"]) == n == d["win"].shape[0] == d["mask"].shape[0]
    assert (d["win"][:, -1] == np.arange(n)).all(), (
        "la ultima posicion de la ventana debe ser el evento puntuado")
    assert int(d["tr"].sum() + d["va"].sum() + d["te"].sum()) == n
    ts = d["df"]["ts"]
    assert ts[d["tr"]].max() <= ts[d["va"]].min()
    assert ts[d["va"]].max() <= ts[d["te"]].min()
    for col in cfg.COLUMNAS_ANALISIS:
        assert col not in d["X_A"].columns
    fa.verificar_sin_orden(d["X_A"])


# --------------------------------------------------------------- modelo A


def correr_a(d: dict, seeds=cfg.SEEDS_MODELO) -> dict:
    """Logistica y LightGBM sobre las semillas. Solo train y validacion."""
    X = d["X_A"].to_numpy()
    y, tr, va = d["y"], d["tr"], d["va"]
    salida: dict = {"logistica": [], "gbm": [], "modelos_gbm": [], "p_val": {}}
    for s in seeds:
        rep.fijar_semillas(s)
        log = ma.entrenar_logistica(X[tr], y[tr], seed=s)
        gbm = ma.entrenar_gbm(X[tr], y[tr], X[va], y[va], seed=s)
        p_log, p_gbm = ma.predecir(log, X[va]), ma.predecir(gbm, X[va])
        salida["logistica"].append(met.auc_pr(y[va], p_log))
        salida["gbm"].append(met.auc_pr(y[va], p_gbm))
        salida["modelos_gbm"].append(gbm)
        salida["p_val"][s] = p_gbm
    return salida


# --------------------------------------------------------------- modelo B / C


def correr_b(d: dict, seed: int, hibrido: bool = False, epocas: int = cfg.EPOCAS_MAX,
             verbose: int = 0) -> dict:
    """Entrena B (o el hibrido C) y devuelve modelo, p_val y epocas usadas."""
    y, tr, va = d["y"], d["tr"], d["va"]
    agg = d["X_A_esc"] if hibrido else None
    d_agg = agg.shape[1] if hibrido else 0

    rep.fijar_semillas(seed)
    lotes_tr = mb.Lotes(d["win"][tr], d["mask"][tr], d["E_num"], d["E_cat"],
                        y=y[tr], X_agg=None if agg is None else agg[tr],
                        batch_size=cfg.BATCH_SIZE, barajar=True)
    lotes_va = mb.Lotes(d["win"][va], d["mask"][va], d["E_num"], d["E_cat"],
                        y=y[va], X_agg=None if agg is None else agg[va],
                        batch_size=cfg.BATCH_SIZE)
    modelo = mb.construir_modelo(d["K"], d["E_num"].shape[1],
                                 fe.cardinalidades(d["vocab"]), d_agg=d_agg)
    t0 = time.time()
    hist = mb.entrenar(modelo, lotes_tr, lotes_va, mb.pesos_de_clase(y[tr]),
                       epocas=epocas, verbose=verbose)
    p_val = mb.predecir(modelo, lotes_va)
    return {
        "modelo": modelo, "p_val": p_val,
        "auc_pr": met.auc_pr(y[va], p_val),
        "epocas": len(hist.history["loss"]),
        "segundos": time.time() - t0,
    }


def correr_b_cacheado(d: dict, seed: int, ruta, hibrido: bool = False,
                      epocas: int = cfg.EPOCAS_MAX, verbose: int = 0) -> dict:
    """Entrena, o recupera del disco un modelo ya entrenado.

    Reejecutar el notebook completo cuesta horas de GRU en CPU. Con los
    pesos en disco cuesta minutos y las cifras son las mismas, porque las
    semillas estan fijas. Borrar el directorio fuerza el reentrenamiento.
    """
    import keras

    ruta = Path(ruta)
    huella_actual = _huella(d, seed, hibrido)
    ruta_huella = ruta.with_suffix(".huella.json")

    if ruta.exists() and ruta_huella.exists():
        guardada = json.loads(ruta_huella.read_text(encoding="utf-8"))
        if guardada == huella_actual:
            modelo = keras.models.load_model(ruta)
            p_val = predecir_split(modelo, d, d["va"], hibrido=hibrido)
            return {"modelo": modelo, "p_val": p_val,
                    "auc_pr": met.auc_pr(d["y"][d["va"]], p_val),
                    "epocas": None, "segundos": 0.0, "desde_cache": True}

    r = correr_b(d, seed, hibrido=hibrido, epocas=epocas, verbose=verbose)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    r["modelo"].save(ruta)
    ruta_huella.write_text(json.dumps(huella_actual, indent=2), encoding="utf-8")
    r["desde_cache"] = False
    return r


def _huella(d: dict, seed: int, hibrido: bool) -> dict:
    """Identifica los datos con los que se entreno un modelo en cache.

    Sin esto, un modelo entrenado en modo dev (400 tarjetas) se cargaria
    en una corrida completa si las formas coincidieran por casualidad, y
    el notebook reportaria cifras de otro experimento sin fallar.
    """
    return {
        "n_eventos": int(len(d["df"])),
        "n_tarjetas": int(d["df"]["card_id"].nunique()),
        "K": int(d["K"]),
        "d_num": int(d["E_num"].shape[1]),
        "cardinalidades": {k: int(v) for k, v in fe.cardinalidades(d["vocab"]).items()},
        "n_train": int(d["tr"].sum()),
        "hibrido": bool(hibrido),
        "d_agg": int(d["X_A_esc"].shape[1]) if hibrido else 0,
        "seed": int(seed),
        "seed_datos": int(cfg.SEED_DATOS),
    }


def predecir_split(modelo, d: dict, sel: np.ndarray, hibrido: bool = False) -> np.ndarray:
    """Puntajes de B/C sobre un subconjunto cualquiera del indice canonico."""
    agg = d["X_A_esc"][sel] if hibrido else None
    lotes = mb.Lotes(d["win"][sel], d["mask"][sel], d["E_num"], d["E_cat"],
                     X_agg=agg, batch_size=cfg.BATCH_SIZE)
    return mb.predecir(modelo, lotes)


# --------------------------------------------- prueba 1: permutacion


def tabla_permutacion(modelo, d: dict, p_a_val: np.ndarray, hibrido: bool = False,
                      semilla_rng: int = 0) -> pd.DataFrame:
    """Reevalua B con los MISMOS pesos sobre ventanas barajadas.

    No se reentrena. A no se recalcula porque sus agregados son invariantes
    a la permutacion por construccion: si se moviera, habria fuga de orden.
    """
    y, va = d["y"], d["va"]
    agg = d["X_A_esc"][va] if hibrido else None
    filas = []
    for modo in ("original", "full", "history"):
        win_eval = (d["win"][va] if modo == "original"
                    else ven.permutar(d["win"][va], d["mask"][va], modo,
                                      np.random.default_rng(semilla_rng)))
        lotes = mb.Lotes(win_eval, d["mask"][va], d["E_num"], d["E_cat"],
                         X_agg=agg, batch_size=cfg.BATCH_SIZE)
        filas.append({
            "variante": modo,
            "auc_pr_B": met.auc_pr(y[va], mb.predecir(modelo, lotes)),
            "auc_pr_A": met.auc_pr(y[va], p_a_val),
        })
    t = pd.DataFrame(filas)
    assert t["auc_pr_A"].nunique() == 1, "BUG: A se movio al permutar"
    t["caida_B"] = t.loc[t["variante"] == "original", "auc_pr_B"].iloc[0] - t["auc_pr_B"]
    return t


# --------------------------------------------- prueba 2: recorte de historia


def curva_k(ks=(1, 3, 5, 10, 20), seed: int = cfg.SEEDS_MODELO[0],
            n_tarjetas: int | None = None, verbose: int = 0,
            dir_cache: Path | None = None) -> pd.DataFrame:
    """AUC-PR de validacion en funcion de cuanta historia ve el modelo.

    Con K=1 el secuencial degenera en un clasificador puntual y deberia
    caer a la altura de A: control de sanidad de la figura.

    Son cinco entrenamientos completos. Con `dir_cache` cada uno queda en
    disco, de modo que una corrida interrumpida retoma donde iba.
    """
    filas = []
    for K in ks:
        d = preparar(n_tarjetas=n_tarjetas, K=K)
        if dir_cache is None:
            r = correr_b(d, seed=seed, verbose=verbose)
        else:
            r = correr_b_cacheado(d, seed=seed, verbose=verbose,
                                  ruta=Path(dir_cache) / f"b_K{K}_semilla{seed}.keras")
        filas.append({"K": K, "auc_pr": r["auc_pr"], "epocas": r["epocas"],
                      "segundos": r["segundos"]})
    return pd.DataFrame(filas)


# --------------------------------------------------------------- cache


def en_cache(ruta, fn, forzar: bool = False):
    """Calcula `fn()` una vez y lo deja en JSON.

    Permite reejecutar el notebook completo sin repetir horas de GRU. Con
    `forzar=True`, o borrando el archivo, se recalcula desde cero.
    """
    ruta = Path(ruta)
    if ruta.exists() and not forzar:
        return json.loads(ruta.read_text(encoding="utf-8"))
    valor = fn()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(valor, indent=2, default=str), encoding="utf-8")
    return valor
