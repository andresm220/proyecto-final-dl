import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import generador as gen
from monitoreo import particion as part


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=200)
    return df, part.asignar_split(df)


def test_cubre_todas_las_filas_una_sola_vez(datos):
    df, s = datos
    assert len(s) == len(df)
    assert set(s.unique()) == {"train", "val", "test"}


def test_proporciones_cercanas_a_70_15_15(datos):
    _, s = datos
    p = s.value_counts(normalize=True)
    assert abs(p["train"] - 0.70) < 0.02
    assert abs(p["val"] - 0.15) < 0.02
    assert abs(p["test"] - 0.15) < 0.02


def test_sin_solape_temporal(datos):
    df, s = datos
    ts = df["ts"]
    assert ts[s == "train"].max() <= ts[s == "val"].min()
    assert ts[s == "val"].max() <= ts[s == "test"].min()


def test_el_corte_es_global_no_por_tarjeta(datos):
    """Una misma tarjeta puede aparecer en train y en test: es realista."""
    df, s = datos
    compartidas = set(df.loc[s == "train", "card_id"]) & set(df.loc[s == "test", "card_id"])
    assert len(compartidas) > 0


def test_no_es_particion_aleatoria(datos):
    """Penalizacion de -20 pts. Una particion aleatoria mezclaria las fechas."""
    df, s = datos
    orden = df["ts"].rank(method="first")
    corte_train = orden[s == "train"].max()
    assert (orden[s != "train"] > corte_train).all()


def test_tabla_reporta_fechas_y_tasa_de_fraude(datos):
    df, s = datos
    t = part.tabla(df, s)
    assert list(t.columns) == ["split", "n", "fecha_min", "fecha_max", "n_fraude", "tasa_fraude"]
    assert list(t["split"]) == ["train", "val", "test"]
    assert (t["tasa_fraude"] > 0).all()
