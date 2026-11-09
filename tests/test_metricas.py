import numpy as np
import pandas as pd
import pytest
from monitoreo import metricas as met


def test_auc_pr_perfecto():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert met.auc_pr(y, p) == pytest.approx(1.0)


def test_auc_pr_aleatorio_cerca_de_la_prevalencia():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.012).astype(int)
    p = rng.random(20000)
    assert abs(met.auc_pr(y, p) - 0.012) < 0.01


def test_en_umbral_cuenta_bien():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    m = met.en_umbral(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_en_umbral_sin_positivos_no_divide_por_cero():
    y = np.array([1, 0])
    p = np.array([0.1, 0.1])
    m = met.en_umbral(y, p, 0.9)
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_desglose_separa_los_subtipos():
    y = np.array([0, 1, 1, 1, 0, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.1, 0.05])
    sub = np.array(["none", "f1_sondeo", "f1_golpe", "f2", "none", "none"])
    t = met.desglose_por_tipo(y, p, sub, 0.5)
    assert set(t["grupo"]) == {"f1_sondeo", "f1_golpe", "f2"}
    assert t.loc[t["grupo"] == "f1_golpe", "recall"].iloc[0] == pytest.approx(1.0)
    assert t.loc[t["grupo"] == "f2", "recall"].iloc[0] == pytest.approx(0.0)


def test_desglose_compara_cada_tipo_contra_los_legitimos():
    """Cada fila mide ese mecanismo vs todo lo legitimo, no vs otros fraudes."""
    y = np.array([0] * 100 + [1] * 5)
    p = np.concatenate([np.full(100, 0.1), np.full(5, 0.9)])
    sub = np.array(["none"] * 100 + ["f3"] * 5)
    t = met.desglose_por_tipo(y, p, sub, 0.5)
    assert t.loc[t["grupo"] == "f3", "n"].iloc[0] == 105


def test_resumen_media_y_sigma():
    m, s = met.resumen([0.50, 0.52, 0.54])
    assert m == pytest.approx(0.52)
    assert s == pytest.approx(np.std([0.50, 0.52, 0.54], ddof=1))


def test_no_expone_exactitud():
    """-15 pts si la exactitud aparece como metrica. No debe existir."""
    assert not hasattr(met, "exactitud")
    assert "accuracy" not in met.en_umbral(np.array([0, 1]), np.array([0.1, 0.9]), 0.5)
