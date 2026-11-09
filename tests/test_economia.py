import numpy as np
import pytest
from monitoreo import config as cfg
from monitoreo import economia as eco


def test_costo_suma_fn_y_fp():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.01, 0.9, 0.01])
    # u=0.5 -> 1 TP, 1 FN, 1 FP, 1 TN
    assert eco.costo(y, p, 0.5) == pytest.approx(cfg.COSTO_FN + cfg.COSTO_FP)


def test_umbral_optimo_de_un_puntaje_calibrado_cae_cerca_del_teorico():
    """p* = 180/4200 = 0.0429. Es la cifra citable de la presentacion."""
    rng = np.random.default_rng(0)
    n = 200000
    p = rng.beta(0.6, 12.0, n)          # puntajes calibrados y desbalanceados
    y = (rng.random(n) < p).astype(int)  # por construccion P(y=1|p)=p
    u, _ = eco.umbral_optimo(y, p)
    assert abs(u - cfg.UMBRAL_TEORICO) < 0.02


def test_el_umbral_optimo_esta_lejos_de_0_5():
    rng = np.random.default_rng(0)
    n = 100000
    p = rng.beta(0.6, 12.0, n)
    y = (rng.random(n) < p).astype(int)
    u, _ = eco.umbral_optimo(y, p)
    assert u < 0.2


def test_curva_devuelve_umbrales_y_costos_alineados():
    y = np.array([1, 0, 1, 0] * 50)
    p = np.linspace(0, 1, 200)
    us, cs = eco.curva(y, p, n_pasos=100)
    assert us.shape == cs.shape == (100,)
    assert (cs >= 0).all()


def test_el_optimo_de_la_curva_es_el_minimo():
    y = np.array([1, 0, 1, 0] * 50)
    p = np.linspace(0, 1, 200)
    us, cs = eco.curva(y, p, n_pasos=1000)
    u, c = eco.umbral_optimo(y, p, n_pasos=1000)
    assert c == pytest.approx(cs.min())
    assert u == pytest.approx(us[cs.argmin()])


def test_ahorro_mensual_escala_por_dias():
    ahorro = eco.ahorro_mensual(costo_a=10000.0, costo_b=7000.0, dias_test=15.0)
    assert ahorro == pytest.approx(3000.0 * 30.0 / 15.0)
