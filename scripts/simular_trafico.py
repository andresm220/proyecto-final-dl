"""Reproduce tráfico real del periodo de prueba contra la API desplegada.

Manda N transacciones consecutivas (en orden cronológico) como lo haría el sistema
autorizador del banco y, después, sus etiquetas reales por /retroalimentacion,
simulando los contracargos que llegan días más tarde. Sirve para:
  - generar registros de uso y evidencia de latencia bajo carga,
  - alimentar el panel de monitoreo con desempeño real del campeón y del retador,
  - con --deriva, demostrar que el monitoreo detecta un cambio en los datos.

    python -m scripts.simular_trafico --url https://<usuario>-<space>.hf.space --n 1500
    python -m scripts.simular_trafico --url http://localhost:7860 --n 300 --deriva 2.5
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from scripts.datos_prueba import cargar, solicitud


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--deriva", type=float, default=1.0,
                    help="Multiplica todos los montos (1.0 = sin deriva).")
    ap.add_argument("--sin-etiquetas", action="store_true")
    a = ap.parse_args()

    print("Regenerando datos del periodo de prueba…", flush=True)
    df, es_test = cargar()
    idx_test = np.flatnonzero(es_test)
    orden = idx_test[np.argsort(df["ts"].to_numpy()[idx_test], kind="mergesort")]
    rng = np.random.default_rng(a.semilla)
    inicio = int(rng.integers(0, max(len(orden) - a.n, 1)))
    elegidos = orden[inicio : inicio + a.n]
    y = df["is_fraud"].to_numpy()[elegidos]
    print(f"{len(elegidos)} transacciones desde {df['ts'].iloc[elegidos[0]]} "
          f"({int(y.sum())} fraudes reales)", flush=True)

    origen = "simulador" if a.deriva == 1.0 else f"simulador-deriva-x{a.deriva:g}"
    cuerpos = [solicitud(df, int(i), factor_monto=a.deriva) for i in elegidos]
    cliente = httpx.Client(base_url=a.url.rstrip("/"), timeout=60,
                           headers={"X-Origen": origen})

    def enviar(cuerpo):
        t = time.perf_counter()
        r = cliente.post("/predecir", json=cuerpo)
        return r, (time.perf_counter() - t) * 1000

    t0 = time.time()
    with ThreadPoolExecutor(a.hilos) as ex:
        resultados = list(ex.map(enviar, cuerpos))
    dur = time.time() - t0

    ok = [(r.json(), ms) for r, ms in resultados if r.status_code == 200]
    errores = [r.status_code for r, _ in resultados if r.status_code != 200]
    ida_vuelta = np.array([ms for _, ms in ok])
    servidor = np.array([j["latencia_ms"] for j, _ in ok])
    dec = [j["decision"] for j, _ in ok]
    print(f"\n{len(ok)} OK, {len(errores)} errores {set(errores) or ''} en {dur:.1f} s "
          f"({len(ok) / dur:.1f} tx/s)")
    print(f"Latencia en servidor   p50 {np.percentile(servidor, 50):.0f} ms · "
          f"p95 {np.percentile(servidor, 95):.0f} ms")
    print(f"Latencia ida y vuelta  p50 {np.percentile(ida_vuelta, 50):.0f} ms · "
          f"p95 {np.percentile(ida_vuelta, 95):.0f} ms")
    print("Decisiones:", {d: dec.count(d) for d in ("allow", "review", "block")})

    if not a.sin_etiquetas:
        print("\nEnviando etiquetas (contracargos)…", flush=True)
        n_et = 0
        for (r, _), etiqueta in zip(resultados, y):
            if r.status_code == 200:
                cliente.post("/retroalimentacion",
                             json={"id": r.json()["id"], "is_fraud": bool(etiqueta)})
                n_et += 1
        print(f"{n_et} etiquetas enviadas. Ver {a.url.rstrip('/')}/monitor")


if __name__ == "__main__":
    main()
