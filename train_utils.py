import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_utils import split_scale
from models import (
    build_rnn_model,
    build_gru_model,
    build_lstm_model,
    build_lstm_attention_model,
    build_transformer_model,
)


# ─────────────────────────────────────────────
# Reproductibilité
# ─────────────────────────────────────────────

def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# ─────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    eps  = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    r2   = float(r2_score(y_true, y_pred))

    return mae, rmse, mape, r2


# ─────────────────────────────────────────────
# Entraînement générique
# ─────────────────────────────────────────────

def train_single_model(model, X_train, y_train, X_val, y_val,
                       X_test, scaler, epochs, batch_size, callbacks):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    return history.history, pred


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def train_models(series_df: pd.DataFrame, window: int, epochs: int,
                 batch_size: int, seed: int,
                 selected_models: list[str] | None = None,
                 progress_callback=None) -> dict:
    """
    Entraîne les modèles sélectionnés et renvoie toutes les métriques + prédictions.

    selected_models : liste de noms parmi
        ["RNN vanilla", "GRU", "LSTM Baseline",
         "LSTM + Attention temporelle", "Transformer lite"]
    Si None → tous les modèles.
    """
    set_seed(seed)

    if selected_models is None:
        selected_models = [
            "RNN vanilla", "GRU", "LSTM Baseline",
            "LSTM + Attention temporelle", "Transformer lite",
        ]

    values = series_df["target"].values.astype(float)

    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     scaler,  test_start_index) = split_scale(values, window)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, verbose=0),
    ]

    histories: dict = {}
    preds:     dict = {}
    y_true = scaler.inverse_transform(y_test)

    # Étapes de progression
    n_models = len(selected_models)
    step = 0.80 / (n_models + 1)   # réserve 20 % pour setup + fin

    def _cb(frac: float, msg: str):
        if progress_callback:
            progress_callback(min(frac, 1.0), msg)

    _cb(0.10, "Données préparées — démarrage de l'entraînement…")

    # ── Baseline naïve (toujours incluse) ──────────────────
    naive_scaled = X_test[:, -1, 0].reshape(-1, 1)
    preds["Naïf"] = scaler.inverse_transform(naive_scaled)

    progress = 0.10

    # ── Boucle sur les modèles demandés ────────────────────
    for name in selected_models:
        progress += step
        _cb(progress, f"Entraînement : {name}…")

        try:
            if name == "RNN vanilla":
                m = build_rnn_model(window)
                histories[name], preds[name] = train_single_model(
                    m, X_train, y_train, X_val, y_val,
                    X_test, scaler, epochs, batch_size, callbacks)

            elif name == "GRU":
                m = build_gru_model(window)
                histories[name], preds[name] = train_single_model(
                    m, X_train, y_train, X_val, y_val,
                    X_test, scaler, epochs, batch_size, callbacks)

            elif name == "LSTM Baseline":
                m = build_lstm_model(window)
                histories[name], preds[name] = train_single_model(
                    m, X_train, y_train, X_val, y_val,
                    X_test, scaler, epochs, batch_size, callbacks)

            elif name == "LSTM + Attention temporelle":
                m, att_m = build_lstm_attention_model(window)
                hist = m.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=callbacks, verbose=0,
                )
                histories[name] = hist.history
                preds[name] = scaler.inverse_transform(
                    m.predict(X_test, verbose=0))
                alpha = att_m.predict(X_test, verbose=0).squeeze(-1)

            elif name == "Transformer lite":
                m = build_transformer_model(window)
                histories[name], preds[name] = train_single_model(
                    m, X_train, y_train, X_val, y_val,
                    X_test, scaler, epochs, batch_size, callbacks)

        except Exception as e:
            _cb(progress, f"⚠️ {name} échoué : {e}")
            continue

    # Si LSTM+Att n'a pas été sélectionné, pas d'alpha
    if "LSTM + Attention temporelle" not in selected_models:
        alpha = np.full((len(X_test), window), 1.0 / window)

    # ── Métriques ──────────────────────────────────────────
    metrics_rows = []
    for name, pred in preds.items():
        mae, rmse, mape, r2 = compute_metrics(y_true, pred)
        metrics_rows.append({
            "Modèle":    name,
            "MAE":       mae,
            "RMSE":      rmse,
            "MAPE (%)":  mape,
            "R²":        r2,
        })

    metrics_df = pd.DataFrame(metrics_rows)

    test_dates = (
        series_df["date"].iloc[test_start_index:]
        .reset_index(drop=True)
    )
    # Aligner la longueur
    n_test = len(y_true)
    test_dates = test_dates.iloc[:n_test].reset_index(drop=True)

    _cb(1.0, "Comparaison terminée ✅")

    return {
        "metrics":          metrics_df,
        "predictions":      preds,
        "y_true":           y_true,
        "histories":        histories,
        "attention":        alpha,
        "test_dates":       test_dates,
        "n_train":          len(X_train),
        "n_val":            len(X_val),
        "n_test":           len(X_test),
        "window":           window,
        "epochs":           epochs,
        "batch_size":       batch_size,
        "seed":             seed,
        "test_start_index": test_start_index,
        "selected_models":  selected_models,
    }