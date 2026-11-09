"""Generador sintetico del Proyecto 1.

Entregable en si mismo: generar(seed) devuelve siempre el mismo DataFrame.
El diseno de f1 esta explicado en la seccion 4.1.1 del spec: los sondeos
escalan de forma monotona para que la senal viva en el ORDEN y no en los
agregados, y se inyectan rafagas legitimas con la misma firma agregada
para que el Modelo A no pueda ganar sin leer secuencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

FECHA_INICIO = pd.Timestamp("2026-01-01")

COLUMNAS = [
    "card_id", "ts", "amount", "merchant_id", "mcc",
    "channel", "country", "is_fraud", "fraud_type", "fraud_subtype",
]

# Distribucion diurna de la hora de compra (suma 1, indices 0..23).
_PESOS_HORA = np.array(
    [0.4, 0.3, 0.2, 0.2, 0.2, 0.4, 1.0, 2.0, 4.0, 5.0, 5.5, 6.0,
     6.5, 6.0, 5.5, 5.5, 6.0, 6.5, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
)
_PESOS_HORA = _PESOS_HORA / _PESOS_HORA.sum()


def _n_tx_por_tarjeta(rng: np.random.Generator, n: int) -> np.ndarray:
    """Lognormal truncada a [TX_MIN, TX_MAX] con media cercana a TX_MEDIA.

    Una uniforme sobre 60-200 daria media 130 y ~520k eventos, fuera del
    objetivo de 400k del spec.
    """
    out = np.empty(n, dtype=np.int32)
    pendientes = np.arange(n)
    while pendientes.size:
        x = rng.lognormal(np.log(95.0), 0.35, size=pendientes.size)
        ok = (x >= cfg.TX_MIN) & (x <= cfg.TX_MAX)
        out[pendientes[ok]] = x[ok].astype(np.int32)
        pendientes = pendientes[~ok]
    return out


def perfiles(rng: np.random.Generator, n_tarjetas: int) -> pd.DataFrame:
    """Un perfil de gasto estable por tarjeta."""
    n_tx = _n_tx_por_tarjeta(rng, n_tarjetas)
    return pd.DataFrame(
        {
            "card_id": np.arange(n_tarjetas, dtype=np.int32),
            "n_tx": n_tx,
            "monto_base": rng.lognormal(np.log(120.0), 0.6, size=n_tarjetas),
            # derivada de n_tx: todas las tarjetas comparten la misma
            # ventana calendario (HORIZONTE_DIAS), asi el timeline global no
            # se adelgaza al final por tarjetas lentas que siguen activas
            # cuando las rapidas ya terminaron
            "tasa_dia": n_tx / cfg.HORIZONTE_DIAS,
            "comercios_pref": [
                rng.choice(cfg.N_COMERCIOS, size=rng.integers(5, 16), replace=False)
                for _ in range(n_tarjetas)
            ],
            "mcc_pref": [
                rng.choice(len(cfg.MCCS), size=rng.integers(3, 8), replace=False)
                for _ in range(n_tarjetas)
            ],
            "pais_base": np.where(rng.random(n_tarjetas) < 0.97, "GT", "US"),
        }
    )


def _timestamps(rng: np.random.Generator, m: int) -> np.ndarray:
    """Fechas de un proceso de Poisson homogeneo condicionado al conteo: m
    ofertas uniformes en [0, HORIZONTE_DIAS) dias, con la hora del dia
    remuestreada de una distribucion diurna. Todas las tarjetas comparten
    la misma ventana calendario, asi el timeline global no se adelgaza al
    final. Se reordena para preservar monotonia."""
    dias = np.sort(rng.uniform(0.0, cfg.HORIZONTE_DIAS, size=m))
    ts = FECHA_INICIO + pd.to_timedelta(dias, unit="D")
    dia = ts.normalize()
    h = rng.choice(24, size=m, p=_PESOS_HORA)
    minuto = rng.integers(0, 60, size=m)
    segundo = rng.integers(0, 60, size=m)
    ts = dia + pd.to_timedelta(h, "h") + pd.to_timedelta(minuto, "m") + pd.to_timedelta(segundo, "s")
    ts = np.sort(ts.values)
    # Remuestrear la hora puede producir colisiones. Se desempatan con
    # microsegundos crecientes: sin esto, drop_duplicates bajaria el conteo
    # de la tarjeta por debajo de TX_MIN y romperia el contrato del generador.
    iguales = np.concatenate([[False], ts[1:] == ts[:-1]])
    if iguales.any():
        ts = ts + np.cumsum(iguales) * np.timedelta64(1, "us")
        ts = np.sort(ts)
    return ts


def flujo_legitimo(rng: np.random.Generator, perf: pd.DataFrame) -> pd.DataFrame:
    """Transacciones normales de todas las tarjetas."""
    trozos = []
    for fila in perf.itertuples(index=False):
        m = int(fila.n_tx)
        ts = _timestamps(rng, m)
        pref = np.asarray(fila.comercios_pref)
        usa_pref = rng.random(m) < 0.8
        comercio = np.where(
            usa_pref,
            rng.choice(pref, size=m),
            rng.integers(0, cfg.N_COMERCIOS, size=m),
        )
        mcc_idx = np.where(
            rng.random(m) < 0.85,
            rng.choice(np.asarray(fila.mcc_pref), size=m),
            rng.integers(0, len(cfg.MCCS), size=m),
        )
        monto = fila.monto_base * rng.lognormal(0.0, 0.55, size=m)
        canal = rng.choice(cfg.CANALES, size=m, p=[0.55, 0.28, 0.10, 0.07])
        pais = np.where(rng.random(m) < 0.98, fila.pais_base, rng.choice(cfg.PAISES, size=m))
        trozos.append(
            pd.DataFrame(
                {
                    "card_id": np.full(m, fila.card_id, dtype=np.int32),
                    "ts": ts,
                    "amount": np.round(monto, 2),
                    "merchant_id": comercio.astype(np.int32),
                    "mcc": np.asarray(cfg.MCCS, dtype=object)[mcc_idx],
                    "channel": canal,
                    "country": pais,
                    "is_fraud": np.zeros(m, dtype=np.int8),
                    "fraud_type": np.full(m, "none", dtype=object),
                    "fraud_subtype": np.full(m, "none", dtype=object),
                }
            )
        )
    df = pd.concat(trozos, ignore_index=True)
    df = df.drop_duplicates(subset=["card_id", "ts"], keep="first")
    df = df.sort_values(["card_id", "ts"], kind="mergesort").reset_index(drop=True)
    return df[COLUMNAS]


def _fila(card_id, ts, amount, merchant, mcc, channel, country, tipo, subtipo):
    return {
        "card_id": np.int32(card_id),
        "ts": pd.Timestamp(ts),
        "amount": round(float(amount), 2),
        "merchant_id": np.int32(merchant),
        "mcc": mcc,
        "channel": channel,
        "country": country,
        "is_fraud": np.int8(0 if tipo == "none" else 1),
        "fraud_type": tipo,
        "fraud_subtype": subtipo,
    }


def _anclas(rng, base, n_episodios, reemplazo=False):
    """Elige (tarjeta, instante) donde insertar episodios."""
    rangos = base.groupby("card_id")["ts"].agg(["min", "max"])
    disponibles = rangos.index.to_numpy()
    # una tarjeta se compromete una sola vez por mecanismo, salvo que no
    # alcancen las tarjetas
    reemplazo = reemplazo or n_episodios > disponibles.size
    cards = rng.choice(disponibles, size=n_episodios, replace=reemplazo)
    u = rng.uniform(0.01, 0.99, size=n_episodios)
    t0 = rangos.loc[cards, "min"].to_numpy() + (
        (rangos.loc[cards, "max"].to_numpy() - rangos.loc[cards, "min"].to_numpy()) * u
    )
    return cards, pd.to_datetime(t0)


def _episodio_pequeno(rng, monotono):
    """3-6 montos chicos en comercios distintos. Si monotono, escalan."""
    n = int(rng.integers(3, 7))
    montos = np.round(np.sort(rng.uniform(5.0, 40.0, size=n)), 2)
    if monotono:
        # el escalamiento tiene que sobrevivir al redondeo a centavos: un
        # empate destruye la unica propiedad de f1 que no es invariante a
        # permutacion. El clamp a 40 va DESPUES del bump y se re-chequea la
        # estricta monotonia para no reintroducir un empate en el techo.
        for i in range(1, n):
            if montos[i] <= montos[i - 1]:
                montos[i] = round(montos[i - 1] + 0.01, 2)
        montos = np.minimum(montos, 40.0)
        # recorrido hacia atras: si el clamp reintrodujo un empate, se baja
        # el elemento anterior, lo que puede propagar el ajuste mas atras
        # todavia (caso de 2+ empates encadenados cerca del techo)
        for i in range(n - 1, 0, -1):
            if montos[i] <= montos[i - 1]:
                montos[i - 1] = round(montos[i] - 0.01, 2)
    else:
        rng.shuffle(montos)
    comercios = rng.choice(cfg.N_COMERCIOS, size=n, replace=False)
    minutos = np.cumsum(rng.uniform(3.0, 20.0, size=n))
    return montos, comercios, minutos


def _inyectar_f1(rng, base, perf, n_episodios):
    cards, t0s = _anclas(rng, base, n_episodios)
    filas = []
    for card, t0 in zip(cards, t0s):
        montos, comercios, minutos = _episodio_pequeno(rng, monotono=True)
        for monto, com, mi in zip(montos, comercios, minutos):
            filas.append(_fila(
                card, t0 + pd.Timedelta(minutes=float(mi)), monto, com,
                rng.choice(cfg.MCCS), rng.choice(("POS", "online")), "GT", "f1", "f1_sondeo",
            ))
        if rng.random() < cfg.PROB_F1_BRECHA_LARGA:
            brecha = pd.Timedelta(hours=float(rng.uniform(26.0, 72.0)))  # caso de fallo esperado
        else:
            brecha = pd.Timedelta(minutes=float(rng.uniform(5.0, 55.0)))
        monto_base = float(perf.loc[perf["card_id"] == card, "monto_base"].iloc[0])
        filas.append(_fila(
            card, t0 + pd.Timedelta(minutes=float(minutos[-1])) + brecha,
            monto_base * rng.uniform(8.0, 30.0), rng.integers(0, cfg.N_COMERCIOS),
            rng.choice(cfg.MCCS), rng.choice(("POS", "online")), "GT", "f1", "f1_golpe",
        ))
    return filas


def _inyectar_rafagas_legitimas(rng, base, perf, n_episodios):
    """Mismo perfil agregado que f1, montos DESORDENADOS, sin golpe fraudulento.
    A veces termina en una compra grande legitima, con la MISMA distribucion
    de canal, pais y monto que el golpe de f1: la unica diferencia entre los
    dos mecanismos debe ser el orden de los montos chicos (spec 4.1.1)."""
    cards, t0s = _anclas(rng, base, n_episodios)
    filas = []
    for card, t0 in zip(cards, t0s):
        montos, comercios, minutos = _episodio_pequeno(rng, monotono=False)
        for monto, com, mi in zip(montos, comercios, minutos):
            filas.append(_fila(
                card, t0 + pd.Timedelta(minutes=float(mi)), monto, com,
                rng.choice(cfg.MCCS), rng.choice(("POS", "online")), "GT", "none", "none",
            ))
        if rng.random() < 0.35:
            monto_base = float(perf.loc[perf["card_id"] == card, "monto_base"].iloc[0])
            filas.append(_fila(
                card, t0 + pd.Timedelta(minutes=float(minutos[-1] + rng.uniform(5, 55))),
                monto_base * rng.uniform(8.0, 30.0), rng.integers(0, cfg.N_COMERCIOS),
                rng.choice(cfg.MCCS), rng.choice(("POS", "online")), "GT", "none", "none",
            ))
    return filas


def _inyectar_f2(rng, base, n_episodios):
    cards, t0s = _anclas(rng, base, n_episodios)
    filas = []
    for card, t0 in zip(cards, t0s):
        n = int(rng.integers(3, 6))
        monto = float(rng.choice((200.0, 500.0, 1000.0, 2000.0)))
        minutos = np.cumsum(rng.uniform(2.0, 9.0, size=n))
        comercio = int(rng.integers(0, cfg.N_COMERCIOS))
        for mi in minutos:
            filas.append(_fila(
                card, t0 + pd.Timedelta(minutes=float(mi)),
                monto * rng.uniform(0.97, 1.03), comercio,
                "transporte", "ATM", "GT", "f2", "f2",
            ))
    return filas


def _inyectar_f3(rng, base, perf, n_episodios):
    """Una sola transaccion extrema y AISLADA: sin actividad de la tarjeta en
    las 6h previas, para que sea estructuralmente distinta del golpe de f1.

    `base` debe incluir ya las rafagas legitimas: una compra grande de rafaga
    puede superar el p99.9 de una tarjeta de gasto bajo, y f3 tiene que quedar
    por encima de TODO lo legitimo de esa tarjeta.
    """
    filas = []
    por_tarjeta = {c: g["ts"].to_numpy() for c, g in base.groupby("card_id")}
    # un solo groupby en vez de un escaneo por episodio: con 400k filas y
    # ~3,600 candidatos, escanear seria ~1.4e9 operaciones
    p999_por_tarjeta = base.groupby("card_id")["amount"].quantile(0.999).to_dict()
    cards, t0s = _anclas(rng, base, n_episodios * 3)
    puestos = 0
    for card, t0 in zip(cards, t0s):
        if puestos >= n_episodios:
            break
        ts = por_tarjeta[card]
        ventana = (ts > np.datetime64(t0 - pd.Timedelta(hours=6))) & (ts <= np.datetime64(t0))
        if ventana.any():
            continue
        p999 = float(p999_por_tarjeta[card])
        filas.append(_fila(
            card, t0, p999 * rng.uniform(1.5, 4.0), rng.integers(0, cfg.N_COMERCIOS),
            rng.choice(cfg.MCCS), rng.choice(("online", "POS")),
            rng.choice(("US", "ES", "CR")), "f3", "f3",
        ))
        puestos += 1
    return filas


def generar(seed: int, n_tarjetas: int | None = None) -> pd.DataFrame:
    """Genera el dataset completo. Identico para la misma semilla."""
    n_tarjetas = cfg.n_tarjetas() if n_tarjetas is None else n_tarjetas
    rng = np.random.default_rng(seed)
    perf = perfiles(rng, n_tarjetas)
    base = flujo_legitimo(rng, perf)

    n_obj = int(len(base) * cfg.TASA_FRAUDE / (1.0 - cfg.TASA_FRAUDE))
    n_f1 = max(1, int(n_obj * cfg.MEZCLA_FRAUDE["f1"] / 5.5))   # ~5.5 tx por episodio
    n_f2 = max(1, int(n_obj * cfg.MEZCLA_FRAUDE["f2"] / 4.0))   # ~4 tx por episodio
    n_f3 = max(1, int(n_obj * cfg.MEZCLA_FRAUDE["f3"]))         # 1 tx por episodio

    filas = []
    filas += _inyectar_f1(rng, base, perf, n_f1)
    rafagas = _inyectar_rafagas_legitimas(rng, base, perf, int(n_f1 * cfg.RAFAGAS_LEGITIMAS_POR_F1))
    filas += rafagas
    filas += _inyectar_f2(rng, base, n_f2)
    # f3 va al final y ve las rafagas: su monto debe superar TODO lo legitimo
    # de la tarjeta, rafagas incluidas.
    base_con_rafagas = pd.concat(
        [base, pd.DataFrame(rafagas, columns=COLUMNAS)], ignore_index=True
    ).sort_values(["card_id", "ts"], kind="mergesort")
    filas += _inyectar_f3(rng, base_con_rafagas, perf, n_f3)

    df = pd.concat([base, pd.DataFrame(filas, columns=COLUMNAS)], ignore_index=True)
    df = df.drop_duplicates(subset=["card_id", "ts"], keep="first")
    df = df.sort_values(["card_id", "ts"], kind="mergesort").reset_index(drop=True)
    df["is_fraud"] = df["is_fraud"].astype(np.int8)
    df["card_id"] = df["card_id"].astype(np.int32)
    df["merchant_id"] = df["merchant_id"].astype(np.int32)
    return df[COLUMNAS]
