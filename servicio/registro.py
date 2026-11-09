"""Registro de cada predicción y de las etiquetas que llegan después.

En un banco la etiqueta real (fraude o no) llega días después, cuando el
tarjetahabiente desconoce el cargo (contracargo). Por eso hay dos tablas: las
predicciones se escriben al puntuar y las etiquetas se unen después por `id`.
Con eso el monitoreo puede medir el desempeño real del campeón y del retador.
"""
from __future__ import annotations

import csv
import io
import os
import sqlite3
import threading
from pathlib import Path


def _ruta_por_defecto() -> Path:
    if os.environ.get("DB_PATH"):
        return Path(os.environ["DB_PATH"])
    # En Hugging Face Spaces el almacenamiento persistente se monta en /data.
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        return Path("/data/registros.db")
    return Path(__file__).resolve().parent / "registros.db"


ESQUEMA = """
CREATE TABLE IF NOT EXISTS predicciones (
    id TEXT PRIMARY KEY,
    recibido_utc TEXT NOT NULL,
    card_id INTEGER,
    ts_transaccion TEXT,
    amount REAL,
    channel TEXT,
    mcc TEXT,
    country TEXT,
    n_historia INTEGER,
    campeon TEXT,
    score_campeon REAL,
    p_campeon REAL,
    decision TEXT,
    retador TEXT,
    score_retador REAL,
    p_retador REAL,
    decision_retador TEXT,
    latencia_ms REAL,
    version_modelo TEXT,
    origen TEXT
);
CREATE TABLE IF NOT EXISTS etiquetas (
    id TEXT PRIMARY KEY REFERENCES predicciones(id),
    is_fraud INTEGER NOT NULL,
    recibido_utc TEXT NOT NULL
);
"""


class Registro:
    def __init__(self, ruta: Path | None = None):
        self.ruta = Path(ruta) if ruta else _ruta_por_defecto()
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conectar() as c:
            c.executescript(ESQUEMA)

    def _conectar(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.ruta, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def guardar_prediccion(self, fila: dict) -> None:
        columnas = ", ".join(fila)
        marcas = ", ".join("?" for _ in fila)
        with self._lock, self._conectar() as c:
            c.execute(f"INSERT INTO predicciones ({columnas}) VALUES ({marcas})",
                      tuple(fila.values()))

    def guardar_etiqueta(self, id_: str, is_fraud: int, recibido: str) -> bool:
        with self._lock, self._conectar() as c:
            existe = c.execute("SELECT 1 FROM predicciones WHERE id = ?", (id_,)).fetchone()
            if not existe:
                return False
            c.execute("INSERT OR REPLACE INTO etiquetas VALUES (?, ?, ?)",
                      (id_, int(is_fraud), recibido))
        return True

    def todas(self) -> list[dict]:
        with self._conectar() as c:
            filas = c.execute(
                "SELECT p.*, e.is_fraud FROM predicciones p "
                "LEFT JOIN etiquetas e ON e.id = p.id ORDER BY p.recibido_utc"
            ).fetchall()
        return [dict(f) for f in filas]

    def ultimas(self, n: int = 20) -> list[dict]:
        with self._conectar() as c:
            filas = c.execute(
                "SELECT p.*, e.is_fraud FROM predicciones p "
                "LEFT JOIN etiquetas e ON e.id = p.id "
                "ORDER BY p.recibido_utc DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(f) for f in filas]

    def como_csv(self) -> str:
        filas = self.todas()
        buf = io.StringIO()
        if filas:
            w = csv.DictWriter(buf, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        return buf.getvalue()

    def borrar_todo(self) -> None:
        with self._lock, self._conectar() as c:
            c.execute("DELETE FROM etiquetas")
            c.execute("DELETE FROM predicciones")
