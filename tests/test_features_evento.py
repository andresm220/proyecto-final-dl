import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import features_evento as fe
from monitoreo import generador as gen
from monitoreo import particion as part


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=150)
    split = part.asignar_split(df)
    es_train = (split == "train").to_numpy()
    vocab = fe.construir_vocabularios(df, es_train)
    E_num, E_cat, scaler = fe.construir(df, vocab, es_train)
    return df, es_train, vocab, E_num, E_cat, scaler


def test_formas_y_tipos(datos):
    df, _, _, E_num, E_cat, _ = datos
    assert E_num.shape == (len(df), len(fe.NOMBRES_NUM))
    assert E_cat.shape == (len(df), 3)
    assert E_num.dtype == np.float32
    assert E_cat.dtype == np.int32


def test_indices_reservados_para_pad_y_unk(datos):
    _, _, vocab, _, E_cat, _ = datos
    for tabla in vocab.values():
        assert min(tabla.values()) >= 2
    assert (E_cat >= cfg.UNK).all()


def test_vocabularios_solo_con_train(datos):
    """Penalizacion de -15 pts. Una categoria exclusiva de test debe caer
    a <UNK>, no ganarse un indice propio."""
    df, es_train, vocab, _, _, _ = datos
    comercios_train = set(df.loc[es_train, "merchant_id"])
    for m in vocab["merchant"]:
        assert m in comercios_train


def test_categoria_nueva_en_test_mapea_a_unk():
    df = gen.generar(21, n_tarjetas=60)
    es_train = np.zeros(len(df), dtype=bool)
    es_train[: len(df) // 2] = True
    vocab = fe.construir_vocabularios(df, es_train)
    df2 = df.copy()
    df2.loc[df2.index[-1], "merchant_id"] = 99999
    _, E_cat, _ = fe.construir(df2, vocab, es_train)
    assert E_cat[-1, 2] == cfg.UNK


def test_scaler_ajustado_solo_en_train(datos):
    """Penalizacion de -15 pts. El test tiene que morder: ajustar con todo el
    dataset debe dar un resultado DISTINTO de ajustar solo con train."""
    df, es_train, vocab, E_num, _, scaler = datos
    # la media de las filas de train debe quedar en ~0 tras escalar
    assert abs(E_num[es_train].mean()) < 0.05
    # ajustar con todo (lo prohibido) produce otros numeros
    todo = np.ones(len(df), dtype=bool)
    E_fuga, _, scaler_fuga = fe.construir(df, vocab, todo)
    assert not np.allclose(scaler.mean_, scaler_fuga.mean_), \
        "el scaler da igual con train que con todo: no se esta ajustando solo en train"
    assert not np.allclose(E_num, E_fuga, atol=1e-4)


def test_scaler_reutilizado_no_reajusta(datos):
    df, es_train, vocab, E_num, _, scaler = datos
    E_otra, _, scaler2 = fe.construir(df, vocab, es_train, scaler=scaler)
    assert scaler2 is scaler
    assert np.allclose(E_num, E_otra, atol=1e-6)


def test_delta_t_presente_y_no_negativo(datos):
    _, _, _, E_num, _, _ = datos
    assert "log_delta_t" in fe.NOMBRES_NUM
    assert "es_primera" in fe.NOMBRES_NUM


def test_ablacion_de_delta_t_elimina_la_columna():
    df = gen.generar(21, n_tarjetas=40)
    es_train = np.ones(len(df), dtype=bool)
    vocab = fe.construir_vocabularios(df, es_train)
    E_con, _, _ = fe.construir(df, vocab, es_train, usar_delta_t=True)
    E_sin, _, _ = fe.construir(df, vocab, es_train, usar_delta_t=False)
    assert E_sin.shape[1] == E_con.shape[1] - 1


def test_delta_t_no_cruza_de_tarjeta():
    df = gen.generar(21, n_tarjetas=40).reset_index(drop=True)
    es_train = np.ones(len(df), dtype=bool)
    vocab = fe.construir_vocabularios(df, es_train)
    E_num, _, _ = fe.construir(df, vocab, es_train)
    i = fe.NOMBRES_NUM.index("es_primera")
    primeras = df.groupby("card_id").head(1).index.to_numpy()
    # tras escalar no vale 1, pero debe ser el valor maximo de la columna
    assert np.isclose(E_num[primeras, i], E_num[:, i].max()).all()


def test_fraud_type_fuera_de_las_features(datos):
    _, _, _, E_num, _, _ = datos
    for col in cfg.COLUMNAS_ANALISIS:
        assert col not in fe.NOMBRES_NUM


def test_construir_rechaza_un_frame_desordenado():
    """Un frame desordenado produciria Delta_t basura sin fallar. Mejor que
    reviente."""
    df = gen.generar(31, n_tarjetas=30)
    es_train = np.ones(len(df), dtype=bool)
    vocab = fe.construir_vocabularios(df, es_train)
    revuelto = df.sample(frac=1.0, random_state=0)
    with pytest.raises(AssertionError):
        fe.construir(revuelto, vocab, es_train)
