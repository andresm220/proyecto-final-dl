"""Linea base sin orden. Una linea base debil invalida toda la comparacion."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAY_LIGHTGBM = True
except ImportError:  # fallback declarado en el spec
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAY_LIGHTGBM = False


def entrenar_logistica(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> Pipeline:
    """Piso de referencia obligatorio."""
    return Pipeline(
        [
            ("escala", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=seed)),
        ]
    ).fit(X_tr, y_tr)


def entrenar_gbm(X_tr, y_tr, X_val, y_val, seed: int):
    """Primario. Early stopping y todo ajuste ocurren sobre VALIDACION."""
    if HAY_LIGHTGBM:
        m = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        m.fit(
            X_tr, y_tr,
            eval_X=X_val, eval_y=y_val, eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return m
    m = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, early_stopping=True,
        validation_fraction=0.15, class_weight="balanced", random_state=seed,
    )
    return m.fit(X_tr, y_tr)


def predecir(modelo, X: np.ndarray) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1].astype(float)
