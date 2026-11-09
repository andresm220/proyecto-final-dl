"""Puntuación de una transacción en vuelo.

Principio de diseño: el servicio NO reimplementa ninguna feature. Llama a las mismas
funciones que usó el entrenamiento (`features_agregadas.construir`,
`features_evento.construir`, `ventanas.construir`) sobre un mini-DataFrame con la
historia de la tarjeta + la transacción nueva, y toma la última fila. Así se evita el
desfase entre entrenamiento y servicio ("training/serving skew"), que es la causa más
común de que un modelo rinda peor en producción que en el notebook.
`tests/test_paridad.py` comprueba que el puntaje del servicio coincide con el del
pipeline por lotes.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from monitoreo import calibracion as cal  # noqa: E402
from monitoreo import config as cfg  # noqa: E402
from monitoreo import features_agregadas as fa  # noqa: E402
from monitoreo import features_evento as fe  # noqa: E402
from monitoreo import ventanas as ven  # noqa: E402

DIR_ARTEFACTOS = Path(os.environ.get("DIR_ARTEFACTOS", RAIZ / "artefactos" / "produccion"))

# La feature de contexto más larga es la media de 7 días: más historia no cambia nada.
HORIZONTE_HISTORIA = pd.Timedelta("7d")
MAX_HISTORIA = 200

DECISIONES = ("allow", "review", "block")

# Nombres legibles para explicar al analista por qué el modelo puntuó alto.
NOMBRES_FEATURES = {
    "amt": "monto de la transacción",
    "amt_mean_24h": "monto medio en 24 h",
    "amt_std_24h": "variabilidad del monto en 24 h",
    "amt_max_24h": "monto máximo en 24 h",
    "n_tx_1h": "transacciones en la última hora",
    "n_tx_24h": "transacciones en 24 h",
    "n_merchants_24h": "comercios distintos en 24 h",
    "amt_ratio_to_mean_7d": "monto vs. su media de 7 días",
    "hour_sin": "hora del día",
    "hour_cos": "hora del día",
    "is_weekend": "fin de semana",
}


# Peso de A en el ensamble E (docs/HIPOTESIS.md, adenda H3). Fijo, no ajustado.
PESO_A = 0.5


@dataclass
class ModeloServido:
    """Un modelo servido: uno o varios componentes + su calibrador y sus umbrales.

    A y C_K3 tienen un solo componente. El ensamble E tiene dos (A y C_K3) y su
    puntaje crudo es el promedio de ambos; después se calibra como cualquier otro.
    """
    rol: str
    familia: str
    componentes: dict  # {"A": LGBMClassifier, "C_K3": keras.Model, ...}
    calibrador: object
    u_estrella: float
    u_revision: float

    @property
    def tiene_a(self) -> bool:
        return "A" in self.componentes


class Puntuador:
    """Carga los artefactos una vez y puntúa transacciones."""

    def __init__(self, directorio: Path = DIR_ARTEFACTOS):
        directorio = Path(directorio)
        self.meta = json.loads((directorio / "metadata.json").read_text(encoding="utf-8"))
        with open(directorio / "preprocesadores.pkl", "rb") as f:
            pre = pickle.load(f)
        self.scaler_eventos = pre["scaler_eventos"]
        self.scaler_agregados = pre["scaler_agregados"]

        vocab = json.loads((directorio / "vocabularios.json").read_text(encoding="utf-8"))
        # JSON guarda las llaves como texto; merchant_id es entero en los datos.
        vocab["merchant"] = {int(k): v for k, v in vocab["merchant"].items()}
        self.vocab = vocab
        self.K = int(self.meta["K"])
        self.version = self.meta["version_modelo"]

        self.modelos: dict[str, ModeloServido] = {}
        for rol, info in self.meta["roles"].items():
            archivos = info.get("archivos") or {info["familia"]: info["archivo"]}
            componentes = {}
            for fam, archivo in archivos.items():
                if fam == "A":
                    with open(directorio / archivo, "rb") as f:
                        componentes[fam] = pickle.load(f)
                else:
                    componentes[fam] = keras.models.load_model(directorio / archivo)
            self.modelos[rol] = ModeloServido(
                rol=rol, familia=info["familia"], componentes=componentes,
                calibrador=pre["calibradores"][rol],
                u_estrella=float(info["u_estrella"]), u_revision=float(info["u_revision"]),
            )

        # Promoción sin reentrenar: si el comité decide que el retador pase a decidir,
        # se reinicia el servicio con PROMOVER_RETADOR=1. Cada modelo conserva su
        # propio calibrador y sus propios umbrales; solo se intercambian los roles.
        self.promovido = os.environ.get("PROMOVER_RETADOR", "").strip() in {"1", "true", "si"}
        if self.promovido:
            m = self.modelos
            m["campeon"], m["retador"] = m["retador"], m["campeon"]
            m["campeon"].rol, m["retador"].rol = "campeon", "retador"
            r = self.meta["roles"]
            r["campeon"], r["retador"] = r["retador"], r["campeon"]
            self.version += "+promovido"
            self.meta["version_modelo"] = self.version
        # Un hilo por modelo. Con 2 vCPU y varias solicitudes simultáneas, dejar que
        # torch y LightGBM abran sus propios hilos por solicitud los pone a competir y
        # triplica la latencia p95 (medido: 252 ms → ver README §7). El paralelismo
        # lo dan las solicitudes concurrentes, no cada predicción.
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
        for m in self.modelos.values():
            if m.tiene_a:
                m.componentes["A"].set_params(n_jobs=1)
        # Calentamiento: la primera inferencia de Keras/torch compila y tarda más.
        self._calentar()

    # ------------------------------------------------------------------ datos

    def _armar_df(self, historial: list[dict], transaccion: dict, card_id: int = 0) -> pd.DataFrame:
        filas = [*historial, transaccion]
        df = pd.DataFrame(filas)
        df["card_id"] = np.int32(card_id)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601").dt.tz_convert(None)
        df["merchant_id"] = df["merchant_id"].astype(np.int32)
        df["amount"] = df["amount"].astype(float)
        nueva_ts = df["ts"].iloc[-1]
        hist = df.iloc[:-1]
        if (hist["ts"] > nueva_ts).any():
            raise ValueError("La transacción a puntuar debe ser posterior a toda la historia.")
        # Se conserva lo de los últimos 7 días (contexto de los agregados de A) y,
        # aunque sean más viejas, las últimas K: el evento más antiguo de la ventana
        # del GRU necesita a su predecesor para calcular delta_t y ratio al anterior.
        hist = hist.sort_values("ts", kind="mergesort")
        reciente = (hist["ts"] >= nueva_ts - HORIZONTE_HISTORIA).to_numpy()
        reciente[-self.K:] = True
        hist = hist[reciente].tail(MAX_HISTORIA)
        df = pd.concat([hist, df.iloc[[-1]]])
        # Orden estable: la transacción nueva queda al final aunque empate en ts.
        df["_orden"] = np.arange(len(df))
        df = df.sort_values(["card_id", "ts", "_orden"], kind="mergesort")
        return df.drop(columns="_orden").reset_index(drop=True)

    def _features(self, df: pd.DataFrame) -> dict:
        X_A = fa.construir(df)
        x_a = X_A.to_numpy()[-1:]
        E_num, E_cat, _ = fe.construir(df, self.vocab, np.zeros(len(df), bool),
                                       scaler=self.scaler_eventos)
        win, mask = ven.construir(df, self.K)
        w = win[-1:]
        entradas_seq = {
            "num": E_num[w], "mcc": E_cat[w, 0], "channel": E_cat[w, 1],
            "merchant": E_cat[w, 2], "mask": mask[-1:],
        }
        return {"X_A": X_A, "x_a": x_a, "seq": entradas_seq}

    # ------------------------------------------------------------------ modelos

    def _componente(self, fam: str, modelo, f: dict) -> float:
        if fam == "A":
            return float(modelo.predict_proba(f["x_a"])[0, 1])
        entradas = dict(f["seq"])
        if fam.startswith("C"):
            entradas["agg"] = self.scaler_agregados.transform(f["x_a"]).astype("float32")
        salida = modelo(entradas, training=False)
        return float(keras.ops.convert_to_numpy(salida).ravel()[0])

    def _puntaje_crudo(self, m: ModeloServido, f: dict) -> tuple[float, dict]:
        partes = {fam: self._componente(fam, mod, f) for fam, mod in m.componentes.items()}
        if len(partes) == 1:
            return next(iter(partes.values())), partes
        return PESO_A * partes["A"] + (1 - PESO_A) * partes["C_K3"], partes

    @staticmethod
    def _decidir(m: ModeloServido, p: float) -> str:
        if p >= m.u_estrella:
            return "block"
        if p >= m.u_revision:
            return "review"
        return "allow"

    def _motivos(self, m: ModeloServido, f: dict, n: int = 3) -> list[dict]:
        """Contribuciones SHAP del componente LightGBM (log-odds).

        En el ensamble explican la mitad A del puntaje; la mitad del GRU se reporta
        aparte como su propio puntaje (`componentes` en la respuesta).
        """
        if not m.tiene_a:
            return []
        contrib = m.componentes["A"].predict(f["x_a"], pred_contrib=True)[0][:-1]
        columnas = list(f["X_A"].columns)
        valores = f["x_a"][0]
        orden = np.argsort(-np.abs(contrib))
        motivos = []
        for i in orden:
            col = columnas[i]
            if col.startswith(("channel_", "mcc_")) and valores[i] == 0:
                continue  # "no es canal X" casi nunca es una explicación útil
            nombre = NOMBRES_FEATURES.get(col, col.replace("channel_", "canal ")
                                          .replace("mcc_", "rubro "))
            motivos.append({"feature": col, "descripcion": nombre,
                            "valor": round(float(valores[i]), 3),
                            "efecto": "sube el riesgo" if contrib[i] > 0 else "baja el riesgo",
                            "contribucion": round(float(contrib[i]), 3)})
            if len(motivos) == n:
                break
        return motivos

    # ------------------------------------------------------------------ API

    def puntuar(self, historial: list[dict], transaccion: dict, card_id: int = 0) -> dict:
        t0 = time.perf_counter()
        df = self._armar_df(historial, transaccion, card_id)
        f = self._features(df)
        salida = {}
        for rol, m in self.modelos.items():
            crudo, partes = self._puntaje_crudo(m, f)
            p = float(cal.aplicar(m.calibrador, np.array([crudo]))[0])
            salida[rol] = {
                "familia": m.familia, "score_crudo": crudo, "risk_score": p,
                "componentes": partes,
                "decision": self._decidir(m, p),
                "umbral_bloqueo": m.u_estrella, "umbral_revision": m.u_revision,
            }
        salida["motivos"] = self._motivos(self.modelos["campeon"], f)
        salida["n_historia_usada"] = int(len(df) - 1)
        salida["latencia_ms"] = (time.perf_counter() - t0) * 1000
        return salida

    def _calentar(self) -> None:
        tx = {"ts": "2026-03-20T12:00:00Z", "amount": 100.0, "merchant_id": 1,
              "mcc": "supermercado", "channel": "POS", "country": "GT"}
        hist = [dict(tx, ts="2026-03-19T12:00:00Z"), dict(tx, ts="2026-03-20T10:00:00Z")]
        for _ in range(2):
            self.puntuar(hist, tx)
