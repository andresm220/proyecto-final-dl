"""Modelo B (GRU) y la variante hibrida de la apuesta C.

Los lotes hacen `gather` sobre la matriz de eventos usando win_idx: nunca
se materializa el tensor [N, K, d], que serian ~1.2 GB.
"""
from __future__ import annotations

import keras
import numpy as np
from keras import layers

from . import config as cfg


class Lotes(keras.utils.PyDataset):
    def __init__(self, win_idx, mask, E_num, E_cat, y=None,
                 batch_size=cfg.BATCH_SIZE, X_agg=None, barajar=False, **kw):
        super().__init__(**kw)
        self.win, self.mask = win_idx, mask
        self.E_num, self.E_cat = E_num, E_cat
        self.y, self.X_agg = y, X_agg
        self.batch_size, self.barajar = batch_size, barajar
        self.orden = np.arange(len(win_idx))

    def __len__(self):
        return int(np.ceil(len(self.win) / self.batch_size))

    def on_epoch_end(self):
        if self.barajar:
            np.random.shuffle(self.orden)

    def __getitem__(self, i):
        sel = self.orden[i * self.batch_size : (i + 1) * self.batch_size]
        w = self.win[sel]
        x = {
            "num": self.E_num[w],
            "mcc": self.E_cat[w, 0],
            "channel": self.E_cat[w, 1],
            "merchant": self.E_cat[w, 2],
            "mask": self.mask[sel],
        }
        if self.X_agg is not None:
            x["agg"] = self.X_agg[sel].astype("float32")
        if self.y is None:
            return x
        return x, self.y[sel].astype("float32")


def construir_modelo(K, d_num, cardinalidades, d_agg=0, unidades=cfg.UNIDADES_GRU):
    """GRU sobre LSTM: menos parametros y con K=20 la memoria larga no aporta."""
    ent = {
        "num": keras.Input((K, d_num), name="num"),
        "mcc": keras.Input((K,), dtype="int32", name="mcc"),
        "channel": keras.Input((K,), dtype="int32", name="channel"),
        "merchant": keras.Input((K,), dtype="int32", name="merchant"),
        "mask": keras.Input((K,), dtype="bool", name="mask"),
    }
    embs = [
        layers.Embedding(cardinalidades[k], cfg.DIM_EMB[k], name=f"emb_{k}")(ent[k])
        for k in ("mcc", "channel", "merchant")
    ]
    x = layers.Concatenate(name="secuencia")([ent["num"], *embs])
    x = layers.GRU(unidades, name="gru")(x, mask=ent["mask"])
    x = layers.Dropout(cfg.DROPOUT)(x)

    if d_agg:
        ent["agg"] = keras.Input((d_agg,), name="agg")
        x = layers.Concatenate(name="hibrido")([x, ent["agg"]])

    x = layers.Dense(32, activation="relu")(x)
    salida = layers.Dense(1, activation="sigmoid", name="p")(x)

    modelo = keras.Model(ent, salida)
    modelo.compile(
        optimizer=keras.optimizers.Adam(cfg.LR),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="auc_pr")],
    )
    return modelo


def entrenar(modelo, lotes_tr, lotes_val, class_weight, epocas=cfg.EPOCAS_MAX, verbose=0):
    """EarlyStopping sobre AUC-PR de VALIDACION. Nunca val_loss, jamas test."""
    parada = keras.callbacks.EarlyStopping(
        monitor="val_auc_pr", mode="max", patience=cfg.PACIENCIA,
        restore_best_weights=True, verbose=verbose,
    )
    return modelo.fit(
        lotes_tr, validation_data=lotes_val, epochs=epocas,
        class_weight=class_weight, callbacks=[parada], verbose=verbose,
    )


def predecir(modelo, lotes) -> np.ndarray:
    return modelo.predict(lotes, verbose=0).ravel().astype(float)


def pesos_de_clase(y) -> dict[int, float]:
    n = len(y)
    pos = float(y.sum())
    return {0: n / (2 * (n - pos)), 1: n / (2 * pos)}
