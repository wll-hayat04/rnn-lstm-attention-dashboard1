import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
# Détection automatique des colonnes date
# ─────────────────────────────────────────────

def detect_date_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in ("date", "time", "jour", "day", "mois", "year", "month")):
            candidates.append(col)
    return candidates


# ─────────────────────────────────────────────
# Préparation de la série
# ─────────────────────────────────────────────

def prepare_series(df: pd.DataFrame, value_col: str,
                   date_col: str | None, transform: str) -> pd.DataFrame:
    data = df.copy()

    # --- dates ---
    if date_col and date_col != "Aucune":
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
            data = data.dropna(subset=[date_col])
            data = data.sort_values(date_col)
            dates = data[date_col].reset_index(drop=True)
        except Exception:
            dates = pd.RangeIndex(start=0, stop=len(data))
    else:
        dates = pd.RangeIndex(start=0, stop=len(data))

    # --- valeurs ---
    values = pd.to_numeric(data[value_col], errors="coerce")

    clean = pd.DataFrame({"date": dates, "value": values}).dropna()

    if len(clean) < 20:
        raise ValueError(
            f"La colonne '{value_col}' ne contient que {len(clean)} valeurs numériques valides. "
            "Vérifie ton fichier CSV."
        )

    # --- transformation ---
    if transform == "Log-rendement":
        mask = clean["value"] > 0
        if mask.sum() < 20:
            raise ValueError(
                "Pas assez de valeurs strictement positives pour calculer les log-rendements."
            )
        clean = clean[mask].copy()
        clean["target"] = np.log(clean["value"] / clean["value"].shift(1))
        clean = clean.dropna()

    elif transform == "Différence":
        clean["target"] = clean["value"].diff()
        clean = clean.dropna()

    else:
        clean["target"] = clean["value"]

    return clean.reset_index(drop=True)


# ─────────────────────────────────────────────
# Fenêtres glissantes
# ─────────────────────────────────────────────

def create_windows(values: np.ndarray, window: int):
    X, y = [], []
    v = np.asarray(values).reshape(-1)

    for i in range(window, len(v)):
        X.append(v[i - window:i])
        y.append(v[i])

    if len(X) == 0:
        raise ValueError(
            f"Série trop courte ({len(v)} pts) pour une fenêtre de {window}. "
            "Réduis la fenêtre ou ajoute plus de données."
        )

    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


# ─────────────────────────────────────────────
# Split temporel + normalisation
# ─────────────────────────────────────────────

def split_scale(series_values: np.ndarray, window: int,
                train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(series_values)
    min_needed = window + 30

    if n < min_needed:
        raise ValueError(
            f"Série trop courte : {n} observations. "
            f"Il en faut au moins {min_needed} pour une fenêtre de {window}."
        )

    n_train = int(train_ratio * n)
    n_val   = int((train_ratio + val_ratio) * n)

    if n_train <= window:
        raise ValueError(
            f"La portion train ({n_train} pts) est trop courte pour la fenêtre ({window})."
        )

    train_raw = series_values[:n_train]
    val_raw   = series_values[n_train - window:n_val]
    test_raw  = series_values[n_val - window:]

    scaler = MinMaxScaler()
    train_sc = scaler.fit_transform(train_raw.reshape(-1, 1)).reshape(-1)
    val_sc   = scaler.transform(val_raw.reshape(-1, 1)).reshape(-1)
    test_sc  = scaler.transform(test_raw.reshape(-1, 1)).reshape(-1)

    X_train, y_train = create_windows(train_sc, window)
    X_val,   y_val   = create_windows(val_sc,   window)
    X_test,  y_test  = create_windows(test_sc,  window)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, n_val


# ─────────────────────────────────────────────
# Analyse par régime de volatilité
# ─────────────────────────────────────────────

def compute_volatility_regimes(series_df: pd.DataFrame,
                               y_true: np.ndarray,
                               preds: dict,
                               test_start_index: int,
                               window_vol: int = 20) -> pd.DataFrame:
    """
    Calcule la volatilité réalisée glissante sur le jeu de test
    et classe chaque observation en régime Faible / Normale / Forte.

    Retourne un DataFrame avec les MAE par régime et par modèle.
    """
    target = series_df["target"].values.astype(float)

    # Volatilité glissante sur la cible complète
    vol = pd.Series(target).rolling(window_vol, min_periods=1).std().values

    # Aligner sur le jeu de test
    test_len = len(y_true)
    vol_test = vol[test_start_index:test_start_index + test_len]

    # Garantir même longueur
    min_len = min(len(vol_test), test_len)
    vol_test = vol_test[:min_len]
    y_true_cut = y_true[:min_len].reshape(-1)

    q33 = np.nanpercentile(vol_test, 33)
    q67 = np.nanpercentile(vol_test, 67)

    regimes = np.where(vol_test <= q33, "Faible",
               np.where(vol_test <= q67, "Normale", "Forte"))

    rows = []
    for name, pred in preds.items():
        pred_cut = pred[:min_len].reshape(-1)
        for regime in ["Faible", "Normale", "Forte"]:
            mask = regimes == regime
            if mask.sum() == 0:
                continue
            mae = np.mean(np.abs(y_true_cut[mask] - pred_cut[mask]))
            rows.append({"Modèle": name, "Régime": regime, "MAE": mae,
                         "N observations": int(mask.sum())})

    return pd.DataFrame(rows)