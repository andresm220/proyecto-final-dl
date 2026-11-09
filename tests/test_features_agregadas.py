import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import features_agregadas as fa
from monitoreo import generador as gen


def _mini(montos, horas, comercios=None):
    n = len(montos)
    comercios = list(range(n)) if comercios is None else comercios
    return pd.DataFrame(
        {
            "card_id": np.zeros(n, dtype=np.int32),
            "ts": [pd.Timestamp("2026-03-01") + pd.Timedelta(hours=h) for h in horas],
            "amount": np.asarray(montos, dtype=float),
            "merchant_id": np.asarray(comercios, dtype=np.int32),
            "mcc": ["ropa"] * n,
            "channel": ["POS"] * n,
            "country": ["GT"] * n,
            "is_fraud": np.zeros(n, dtype=np.int8),
            "fraud_type": ["none"] * n,
            "fraud_subtype": ["none"] * n,
        }
    )


def test_closed_left_excluye_la_transaccion_actual():
    df = _mini([10, 20, 30], [0, 1, 2])
    X = fa.construir(df)
    # en la fila 2, el contexto son 10 y 20 -> media 15, no 20
    assert X.loc[2, "amt_mean_24h"] == pytest.approx(15.0)
    assert X.loc[2, "amt_max_24h"] == pytest.approx(20.0)
    assert X.loc[2, "amt"] == pytest.approx(30.0)


def test_primera_transaccion_sin_contexto():
    df = _mini([10, 20], [0, 1])
    X = fa.construir(df)
    assert X.loc[0, "n_tx_24h"] == 0
    assert X.loc[0, "n_tx_1h"] == 0
    # sin contexto los agregados quedan en 0, nunca en NaN: el modelo no
    # puede recibir NaN y un 0 aqui significa "no habia historia"
    assert X.loc[0, "amt_mean_24h"] == 0.0
    assert X.loc[0, "amt_max_24h"] == 0.0
    assert not X.isna().any().any()


def test_envenenamiento_del_futuro_no_altera_el_pasado():
    """El test que atrapa cualquier groupby().mean() sobre el historico
    completo. Penalizacion de -15 pts."""
    df = _mini([10, 20, 30], [0, 1, 2])
    X_antes = fa.construir(df)
    df2 = _mini([10, 20, 30, 999999], [0, 1, 2, 3])
    X_despues = fa.construir(df2)
    pd.testing.assert_frame_equal(X_antes, X_despues.iloc[:3])


def test_conteos_respetan_la_ventana_temporal():
    df = _mini([10, 10, 10, 10], [0, 0.5, 0.75, 30])
    X = fa.construir(df)
    assert X.loc[2, "n_tx_1h"] == 2      # las dos previas dentro de 1h
    assert X.loc[3, "n_tx_24h"] == 0     # 30h despues, nada en la ventana


def test_n_merchants_cuenta_distintos_no_totales():
    df = _mini([10, 10, 10], [0, 1, 2], comercios=[5, 5, 7])
    X = fa.construir(df)
    assert X.loc[2, "n_merchants_24h"] == 1   # solo el comercio 5 esta antes
    df2 = _mini([10, 10, 10], [0, 1, 2], comercios=[5, 6, 7])
    assert fa.construir(df2).loc[2, "n_merchants_24h"] == 2


def test_ninguna_feature_codifica_orden():
    df = gen.generar(11, n_tarjetas=30)
    X = fa.construir(df)
    fa.verificar_sin_orden(X)   # no debe lanzar
    for col in X.columns:
        for patron in cfg.PATRONES_PROHIBIDOS_EN_A:
            assert patron not in col.lower()


def test_los_agregados_son_invariantes_a_permutar_los_montos_del_conjunto():
    """Media, maximo y conteo de un conjunto no dependen del orden.
    Es la razon por la que A no puede moverse en la prueba de permutacion."""
    df = _mini([10, 20, 30, 40], [0, 1, 2, 3])
    X = fa.construir(df)
    df_perm = _mini([30, 10, 20, 40], [0, 1, 2, 3])
    X_perm = fa.construir(df_perm)
    assert X.loc[3, "amt_mean_24h"] == pytest.approx(X_perm.loc[3, "amt_mean_24h"])
    assert X.loc[3, "amt_max_24h"] == pytest.approx(X_perm.loc[3, "amt_max_24h"])
    assert X.loc[3, "n_tx_24h"] == X_perm.loc[3, "n_tx_24h"]


def test_fraud_type_no_aparece_en_las_features():
    df = gen.generar(11, n_tarjetas=30)
    X = fa.construir(df)
    for col in cfg.COLUMNAS_ANALISIS:
        assert col not in X.columns
    assert X.select_dtypes(exclude="number").empty


def test_alineado_con_el_dataframe_de_entrada():
    df = gen.generar(11, n_tarjetas=30)
    X = fa.construir(df)
    assert len(X) == len(df)
    assert (X.index == df.index).all()


def test_las_dos_ventanas_de_24h_coinciden_en_el_borde():
    """n_tx_24h usa rolling(closed='left') y n_merchants_24h un barrido de dos
    punteros. Ambos describen la MISMA ventana, asi que una transaccion que cae
    exactamente sobre el borde izquierdo tiene que entrar en los dos o en
    ninguno."""
    df = _mini([10, 20], [0, 24], comercios=[5, 7])
    X = fa.construir(df)
    assert X.loc[1, "n_tx_24h"] == 1
    assert X.loc[1, "n_merchants_24h"] == 1

    # justo fuera de la ventana: ninguno de los dos debe contarla
    df = _mini([10, 20], [0, 24.001], comercios=[5, 7])
    X = fa.construir(df)
    assert X.loc[1, "n_tx_24h"] == 0
    assert X.loc[1, "n_merchants_24h"] == 0
