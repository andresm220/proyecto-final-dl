"""El monitoreo debe callar cuando no pasa nada y alertar cuando los datos cambian."""
import numpy as np

from servicio.supervision import psi, psi_categorico


def _referencia(rng):
    x = rng.lognormal(5, 1, 20_000)
    cortes = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    conteo, _ = np.histogram(x, bins=cortes)
    return cortes.tolist(), (conteo / conteo.sum()).tolist()


def test_psi_bajo_si_la_distribucion_no_cambia():
    rng = np.random.default_rng(1)
    cortes, esperado = _referencia(rng)
    v, _ = psi(rng.lognormal(5, 1, 2_000), cortes, esperado)
    assert v < 0.05


def test_psi_alto_si_los_montos_se_inflan():
    rng = np.random.default_rng(2)
    cortes, esperado = _referencia(rng)
    v, _ = psi(rng.lognormal(5, 1, 2_000) * 2.5, cortes, esperado)
    assert v > 0.25


def test_valores_fuera_del_rango_caen_en_los_extremos():
    v, actual = psi(np.array([-1e9, 1e9]), [0, 1, 2], [0.5, 0.5])
    assert actual == [0.5, 0.5] and v < 1e-9


def test_psi_categorico_detecta_mezcla_nueva():
    esperado = {"POS": 0.7, "online": 0.3}
    assert psi_categorico(["POS"] * 70 + ["online"] * 30, esperado) < 1e-9
    assert psi_categorico(["ATM"] * 50 + ["POS"] * 50, esperado) > 0.25
