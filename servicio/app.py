"""API de puntuación de riesgo transaccional — Banco del Altiplano.

    uvicorn servicio.app:app --host 0.0.0.0 --port 7860

Rutas:
    GET  /               demo para el analista (HTML)
    GET  /monitor        panel de monitoreo (HTML)
    GET  /docs           documentación interactiva (Swagger, generada por FastAPI)
    GET  /salud          estado del servicio
    GET  /modelo         ficha del modelo: roles, umbrales, métricas
    POST /predecir       puntúa una transacción en vuelo
    POST /retroalimentacion   etiqueta real que llega después (contracargo)
    GET  /metricas       monitoreo en JSON
    GET  /registros.csv  exporta el registro de uso
    GET  /escenarios     casos de ejemplo del periodo de prueba
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .inferencia import DIR_ARTEFACTOS, Puntuador  # agrega src/ al path
from .registro import Registro
from . import supervision as mon

from monitoreo import config as cfg  # noqa: E402

AQUI = Path(__file__).resolve().parent
log = logging.getLogger("servicio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ESTADO: dict = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    ESTADO["puntuador"] = Puntuador()
    ESTADO["registro"] = Registro()
    ESTADO["referencia"] = json.loads(
        (DIR_ARTEFACTOS / "referencia_monitoreo.json").read_text(encoding="utf-8"))
    if ESTADO["puntuador"].promovido:
        ref = ESTADO["referencia"]["puntajes"]
        ref["campeon"], ref["retador"] = ref["retador"], ref["campeon"]
    ESTADO["inicio"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    log.info("Modelo %s cargado. Registro en %s",
             ESTADO["puntuador"].version, ESTADO["registro"].ruta)
    yield


app = FastAPI(
    title="Monitoreo transaccional — API de riesgo",
    version="1.0.0",
    description=(
        "Puntúa una transacción con tarjeta a partir de la historia reciente de la tarjeta. "
        "Devuelve probabilidad calibrada de fraude y la decisión **block / review / allow**, "
        "tomada con un umbral que minimiza el costo esperado (FN = Q4,200; FP = Q180). "
        "El modelo campeón decide; el retador puntúa en modo sombra y solo se registra."
    ),
    lifespan=ciclo_de_vida,
)
app.mount("/static", StaticFiles(directory=AQUI / "static"), name="static")


# ---------------------------------------------------------------- esquemas

class Transaccion(BaseModel):
    ts: dt.datetime = Field(..., description="Fecha y hora de la transacción (ISO 8601).",
                            examples=["2026-03-20T14:31:00Z"])
    amount: float = Field(..., gt=0, le=1_000_000, description="Monto en quetzales.")
    merchant_id: int = Field(..., ge=0, description="Identificador del comercio.")
    mcc: Literal[cfg.MCCS] = Field(..., description="Rubro del comercio.")  # type: ignore[valid-type]
    channel: Literal[cfg.CANALES] = Field(..., description="Canal.")  # type: ignore[valid-type]
    country: str = Field("GT", min_length=2, max_length=2)

    @field_validator("ts")
    @classmethod
    def _con_zona(cls, v: dt.datetime) -> dt.datetime:
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)

    def a_dict(self) -> dict:
        d = self.model_dump()
        d["ts"] = self.ts.isoformat()
        return d


class Solicitud(BaseModel):
    card_id: int = Field(0, ge=0, description="Identificador de la tarjeta (solo para registro).")
    historial: list[Transaccion] = Field(
        default_factory=list, max_length=500,
        description="Transacciones previas de la tarjeta. Mínimo útil: últimos 7 días.")
    transaccion: Transaccion = Field(..., description="Transacción en vuelo que se puntúa.")


class Retroalimentacion(BaseModel):
    id: str
    is_fraud: bool


class ResultadoModelo(BaseModel):
    familia: str
    risk_score: float
    decision: Literal["allow", "review", "block"]


class Respuesta(BaseModel):
    id: str
    risk_score: float = Field(..., description="Probabilidad calibrada de fraude (campeón).")
    decision: Literal["allow", "review", "block"]
    umbral_bloqueo: float
    umbral_revision: float
    model_version: str
    motivos: list[dict]
    retador: ResultadoModelo
    n_historia_usada: int
    latencia_ms: float


# ---------------------------------------------------------------- rutas

@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(AQUI / "static" / "index.html")


@app.get("/monitor", include_in_schema=False)
def panel():
    return FileResponse(AQUI / "static" / "monitor.html")


@app.get("/salud")
def salud():
    p = ESTADO["puntuador"]
    return {"estado": "ok", "model_version": p.version, "desde": ESTADO["inicio"],
            "campeon": p.modelos["campeon"].familia, "retador": p.modelos["retador"].familia}


@app.get("/modelo")
def ficha_modelo():
    m = ESTADO["puntuador"].meta
    return {k: m[k] for k in ("version_modelo", "K", "costos", "capacidad_revision", "roles",
                              "validacion", "hipotesis", "test", "test_info",
                              "permutacion_val", "versiones", "hipotesis_sha256",
                              "inicio_entrenamiento", "fecha_evaluacion_test")}


@app.post("/predecir", response_model=Respuesta)
def predecir(sol: Solicitud, x_origen: str | None = Header(default=None)):
    p: Puntuador = ESTADO["puntuador"]
    try:
        r = p.puntuar([t.a_dict() for t in sol.historial], sol.transaccion.a_dict(), sol.card_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    id_ = uuid.uuid4().hex
    c, ret = r["campeon"], r["retador"]
    tx = sol.transaccion
    ESTADO["registro"].guardar_prediccion({
        "id": id_,
        "recibido_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "card_id": sol.card_id, "ts_transaccion": tx.ts.isoformat(),
        "amount": tx.amount, "channel": tx.channel, "mcc": tx.mcc, "country": tx.country,
        "n_historia": r["n_historia_usada"],
        "campeon": c["familia"], "score_campeon": c["score_crudo"],
        "p_campeon": c["risk_score"], "decision": c["decision"],
        "retador": ret["familia"], "score_retador": ret["score_crudo"],
        "p_retador": ret["risk_score"], "decision_retador": ret["decision"],
        "latencia_ms": r["latencia_ms"], "version_modelo": p.version,
        "origen": (x_origen or "api")[:30],
    })
    return Respuesta(
        id=id_, risk_score=c["risk_score"], decision=c["decision"],
        umbral_bloqueo=c["umbral_bloqueo"], umbral_revision=c["umbral_revision"],
        model_version=p.version, motivos=r["motivos"],
        retador=ResultadoModelo(familia=ret["familia"], risk_score=ret["risk_score"],
                                decision=ret["decision"]),
        n_historia_usada=r["n_historia_usada"], latencia_ms=r["latencia_ms"],
    )


@app.post("/retroalimentacion")
def retroalimentacion(r: Retroalimentacion):
    ok = ESTADO["registro"].guardar_etiqueta(
        r.id, int(r.is_fraud), dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    if not ok:
        raise HTTPException(status_code=404, detail="No existe una predicción con ese id.")
    return {"ok": True}


@app.get("/metricas")
def metricas(origen: str | None = None, ventana: int = 1000):
    """Monitoreo sobre las últimas `ventana` predicciones (0 = todas).

    Una ventana móvil evita que un cambio reciente se diluya entre semanas de
    tráfico normal: la deriva tiene que verse cuando ocurre.
    """
    filas = ESTADO["registro"].todas()
    origenes = sorted({f["origen"] for f in filas})
    if origen:
        filas = [f for f in filas if f["origen"] == origen]
    if ventana > 0:
        filas = filas[-ventana:]
    r = mon.resumen(filas, ESTADO["referencia"], ESTADO["puntuador"].meta)
    r["origenes"], r["ventana"] = origenes, ventana
    return r


@app.get("/registros.csv", response_class=PlainTextResponse)
def registros_csv():
    return PlainTextResponse(ESTADO["registro"].como_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=registros.csv"})


@app.get("/escenarios")
def escenarios():
    ruta = AQUI / "escenarios.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []


@app.delete("/registros", include_in_schema=False)
def borrar_registros(x_admin_token: str | None = Header(default=None)):
    token = os.environ.get("ADMIN_TOKEN")
    if not token or x_admin_token != token:
        raise HTTPException(status_code=403, detail="No autorizado.")
    ESTADO["registro"].borrar_todo()
    return {"ok": True}
