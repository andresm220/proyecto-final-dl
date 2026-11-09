import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import generador as gen
from monitoreo import ventanas as ven


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=120)
    win, mask = ven.construir(df, K=cfg.K)
    return df, win, mask


def test_formas_y_tipos(datos):
    df, win, mask = datos
    assert win.shape == (len(df), cfg.K)
    assert mask.shape == (len(df), cfg.K)
    assert win.dtype == np.int32
    assert mask.dtype == bool


def test_la_ultima_posicion_es_la_propia_transaccion(datos):
    df, win, mask = datos
    assert (win[:, -1] == np.arange(len(df))).all()
    assert mask[:, -1].all()


def test_ninguna_ventana_mira_al_futuro(datos):
    df, win, mask = datos
    ts = df["ts"].to_numpy()
    assert (ts[win][mask] <= np.repeat(ts[:, None], cfg.K, axis=1)[mask]).all()


def test_ninguna_ventana_cruza_de_tarjeta(datos):
    df, win, mask = datos
    card = df["card_id"].to_numpy()
    propia = np.repeat(card[:, None], cfg.K, axis=1)
    assert (card[win][mask] == propia[mask]).all()


def test_la_ventana_es_contigua_y_ordenada(datos):
    df, win, mask = datos
    ts = df["ts"].to_numpy()
    for i in np.random.default_rng(0).choice(len(df), size=200, replace=False):
        validos = ts[win[i]][mask[i]]
        assert (np.diff(validos) >= np.timedelta64(0)).all()


def test_padding_al_inicio_para_historia_corta():
    df = gen.generar(5, n_tarjetas=10)
    win, mask = ven.construir(df, K=cfg.K)
    primera_de_cada_tarjeta = df.reset_index(drop=True).groupby("card_id").head(1).index
    for i in primera_de_cada_tarjeta:
        assert mask[i].sum() == 1
        assert mask[i, -1]
        assert not mask[i, :-1].any()


def test_mascara_cuenta_exactamente_la_historia_disponible(datos):
    df, win, mask = datos
    pos = df.reset_index(drop=True).groupby("card_id").cumcount().to_numpy()
    esperado = np.minimum(pos + 1, cfg.K)
    assert (mask.sum(axis=1) == esperado).all()


def test_memoria_razonable(datos):
    _, win, mask = datos
    assert win.nbytes / len(win) == cfg.K * 4


def test_construir_rechaza_un_frame_desordenado():
    """cumcount depende del orden por (card_id, ts)."""
    df = gen.generar(31, n_tarjetas=30)
    revuelto = df.sample(frac=1.0, random_state=0)
    with pytest.raises(AssertionError):
        ven.construir(revuelto, cfg.K)
