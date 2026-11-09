"""Paridad entrenamiento ↔ servicio.

El riesgo número uno al pasar un modelo a producción es que el servicio calcule las
features distinto que el notebook. Esta prueba toma transacciones reales del periodo
de prueba, las puntúa de dos formas y exige que coincidan:

  - por lotes: el pipeline de entrenamiento sobre el dataset completo;
  - en línea: el servicio, recibiendo solo la historia de 7 días de la tarjeta.
"""
import numpy as np
import pytest

from monitoreo import features_agregadas as fa
from monitoreo import features_evento as fe
from monitoreo import modelos_b as mb
from monitoreo import ventanas as ven
from scripts.datos_prueba import cargar, solicitud
from servicio.inferencia import PESO_A, Puntuador

pytestmark = pytest.mark.lento


@pytest.fixture(scope="module")
def entorno():
    p = Puntuador()
    df, es_test = cargar()
    y = df["is_fraud"].to_numpy()
    rng = np.random.default_rng(0)
    idx_test = np.flatnonzero(es_test)
    fraudes = idx_test[y[idx_test] == 1]
    muestra = np.concatenate([rng.choice(idx_test, 25, replace=False),
                              rng.choice(fraudes, 15, replace=False)])

    X_A = fa.construir(df)
    E_num, E_cat, _ = fe.construir(df, p.vocab, np.zeros(len(df), bool), scaler=p.scaler_eventos)
    win, mask = ven.construir(df, p.K)
    lotes = {}
    for rol, m in p.modelos.items():
        partes = {}
        for fam, mod in m.componentes.items():
            if fam == "A":
                partes[fam] = mod.predict_proba(X_A.to_numpy()[muestra])[:, 1]
            else:
                agg = (p.scaler_agregados.transform(X_A.to_numpy()[muestra]).astype("float32")
                       if fam.startswith("C") else None)
                partes[fam] = mb.predecir(mod, mb.Lotes(win[muestra], mask[muestra],
                                                        E_num, E_cat, X_agg=agg))
        lotes[rol] = (next(iter(partes.values())) if len(partes) == 1
                      else PESO_A * partes["A"] + (1 - PESO_A) * partes["C_K3"])
    return p, df, muestra, lotes


def test_servicio_reproduce_el_puntaje_por_lotes(entorno):
    p, df, muestra, lotes = entorno
    for j, i in enumerate(muestra):
        s = solicitud(df, int(i))
        r = p.puntuar(s["historial"], s["transaccion"], s["card_id"])
        for rol in p.modelos:
            assert r[rol]["score_crudo"] == pytest.approx(lotes[rol][j], abs=1e-5), (
                f"{rol}: evento {i} difiere entre servicio y lotes")


def test_recortar_historia_vieja_no_cambia_el_puntaje(entorno):
    """El contrato pide 7 días de historia: mandar más no debe cambiar nada."""
    p, df, muestra, _ = entorno
    i = int(muestra[0])
    tarjeta = df[df["card_id"] == df.iloc[i]["card_id"]]
    previas = tarjeta[tarjeta.index < df.index[i]]
    from scripts.datos_prueba import _tx
    completa = [_tx(r) for _, r in previas.iterrows()]
    s = solicitud(df, i)
    a = p.puntuar(completa, s["transaccion"])
    b = p.puntuar(s["historial"], s["transaccion"])
    for rol in p.modelos:
        assert a[rol]["score_crudo"] == pytest.approx(b[rol]["score_crudo"], abs=1e-6)
