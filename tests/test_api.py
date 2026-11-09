"""Contrato de la API: entradas válidas, rechazos, registro y monitoreo."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    import os
    os.environ["DB_PATH"] = str(tmp_path_factory.mktemp("db") / "registros.db")
    from servicio.app import app
    with TestClient(app) as c:
        yield c


TX = {"ts": "2026-03-20T14:30:00Z", "amount": 250.0, "merchant_id": 12,
      "mcc": "supermercado", "channel": "POS", "country": "GT"}
HIST = [dict(TX, ts=f"2026-03-{d:02d}T10:00:00Z", amount=100.0 + d) for d in range(14, 20)]


def test_salud(cliente):
    r = cliente.get("/salud").json()
    assert r["estado"] == "ok" and r["model_version"]


def test_prediccion_valida(cliente):
    r = cliente.post("/predecir", json={"card_id": 5, "historial": HIST, "transaccion": TX})
    assert r.status_code == 200
    j = r.json()
    assert 0.0 <= j["risk_score"] <= 1.0
    assert j["decision"] in {"allow", "review", "block"}
    assert j["umbral_revision"] <= j["umbral_bloqueo"]
    assert j["retador"]["decision"] in {"allow", "review", "block"}
    assert j["n_historia_usada"] == len(HIST)


def test_decision_es_coherente_con_umbrales(cliente):
    j = cliente.post("/predecir", json={"historial": HIST, "transaccion": TX}).json()
    p, ur, ub = j["risk_score"], j["umbral_revision"], j["umbral_bloqueo"]
    esperado = "block" if p >= ub else "review" if p >= ur else "allow"
    assert j["decision"] == esperado


def test_sin_historia_tambien_puntua(cliente):
    assert cliente.post("/predecir", json={"transaccion": TX}).status_code == 200


@pytest.mark.parametrize("cambio", [
    {"channel": "telepatia"}, {"mcc": "casino"}, {"amount": -5}, {"amount": 0},
    {"ts": "no-es-fecha"}, {"merchant_id": -1},
])
def test_entradas_invalidas_se_rechazan(cliente, cambio):
    r = cliente.post("/predecir", json={"historial": HIST, "transaccion": dict(TX, **cambio)})
    assert r.status_code == 422


def test_transaccion_anterior_a_la_historia_se_rechaza(cliente):
    viejo = dict(TX, ts="2026-03-01T00:00:00Z")
    r = cliente.post("/predecir", json={"historial": HIST, "transaccion": viejo})
    assert r.status_code == 422


def test_retroalimentacion_y_metricas(cliente):
    j = cliente.post("/predecir", json={"historial": HIST, "transaccion": TX}).json()
    assert cliente.post("/retroalimentacion", json={"id": j["id"], "is_fraud": False}).status_code == 200
    assert cliente.post("/retroalimentacion", json={"id": "no-existe", "is_fraud": True}).status_code == 404
    m = cliente.get("/metricas").json()
    assert m["n_predicciones"] >= 1 and m["n_etiquetadas"] >= 1
    assert "desempeno" in m


def test_registro_exportable(cliente):
    csv = cliente.get("/registros.csv").text
    assert csv.splitlines()[0].startswith("id,")


def test_escenarios_y_paginas(cliente):
    assert isinstance(cliente.get("/escenarios").json(), list)
    assert cliente.get("/").status_code == 200
    assert cliente.get("/monitor").status_code == 200


def test_borrar_registros_requiere_token(cliente):
    assert cliente.delete("/registros").status_code == 403


def test_promocion_del_retador_intercambia_roles(monkeypatch):
    """PROMOVER_RETADOR=1 (configuración del Dockerfile) hace que E decida y A quede en sombra."""
    from servicio.inferencia import Puntuador
    normal = Puntuador()
    monkeypatch.setenv("PROMOVER_RETADOR", "1")
    promovido = Puntuador()
    assert promovido.promovido and promovido.version.endswith("+promovido")
    assert promovido.modelos["campeon"].familia == normal.modelos["retador"].familia
    assert promovido.modelos["retador"].familia == normal.modelos["campeon"].familia
    r = promovido.puntuar(HIST, TX)
    if len(promovido.modelos["campeon"].componentes) > 1:
        assert set(r["campeon"]["componentes"]) == {"A", "C_K3"}
        assert r["motivos"], "el ensamble explica su componente LightGBM"
