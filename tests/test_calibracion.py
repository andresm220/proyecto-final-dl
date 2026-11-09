import numpy as np
import pytest
from monitoreo import calibracion as cal


def test_calibra_puntajes_inflados():
    """Una red con class_weight sobreestima la probabilidad de fraude."""
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.02).astype(int)
    p_crudo = np.clip(np.where(y == 1, rng.beta(6, 3, 20000), rng.beta(2, 4, 20000)), 0, 1)
    c = cal.ajustar(p_crudo, y)
    p_cal = cal.aplicar(c, p_crudo)
    assert abs(p_cal.mean() - y.mean()) < abs(p_crudo.mean() - y.mean())


def test_preserva_el_ranking():
    """La isotonica es monotona no decreciente: no invierte ningun par.

    No conserva el AUC-PR: al aplanar tramos crea empates, y la precision
    promedio si cambia con los empates (aqui 5000 puntajes distintos
    colapsan a 5 mesetas). Por eso el informe reporta AUC-PR sobre el
    puntaje CRUDO -- es una metrica de ranking -- y reserva el calibrado
    para la decision de costo, que es donde la probabilidad debe ser
    interpretable en quetzales.
    """
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(y * 0.4 + rng.random(5000) * 0.6, 0, 1)
    q = cal.aplicar(cal.ajustar(p, y), p)
    orden = np.argsort(p, kind="mergesort")
    assert np.all(np.diff(q[orden]) >= -1e-12)


def test_no_inventa_probabilidades_fuera_del_rango_visto():
    """out_of_bounds='clip': un puntaje mayor a todo lo visto en validacion
    no puede mapear por encima del maximo aprendido."""
    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.05).astype(int)
    p = rng.random(3000) * 0.6
    c = cal.ajustar(p, y)
    assert cal.aplicar(c, np.array([5.0]))[0] <= cal.aplicar(c, np.array([0.6]))[0] + 1e-12


def test_salida_en_cero_uno():
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.1).astype(int)
    p = rng.random(2000)
    q = cal.aplicar(cal.ajustar(p, y), rng.random(500))
    assert (q >= 0).all() and (q <= 1).all()
