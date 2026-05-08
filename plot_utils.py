import io
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "figure.dpi":        110,
    "font.size":         9,
})

# Palette cohérente
PALETTE = {
    "Naïf":                      "#6b7280",
    "RNN vanilla":               "#f59e0b",
    "GRU":                       "#3b82f6",
    "LSTM Baseline":             "#8b5cf6",
    "LSTM + Attention temporelle": "#10b981",
    "Transformer lite":          "#ef4444",
    "Réel":                      "#e5e7eb",
}


def _color(name: str) -> str:
    return PALETTE.get(name, "#94a3b8")


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────
# Série originale
# ─────────────────────────────────────────────

def plot_original_series(series_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(series_df["date"], series_df["value"],
            color="#3b82f6", linewidth=1.4, alpha=0.9)
    ax.fill_between(series_df["date"], series_df["value"],
                    alpha=0.12, color="#3b82f6")
    _style(ax, "Série originale importée", "Temps", "Valeur")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Série transformée
# ─────────────────────────────────────────────

def plot_prepared_series(series_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(series_df["date"], series_df["target"],
            color="#10b981", linewidth=1.2, alpha=0.9)
    ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
    _style(ax, "Série cible après transformation", "Temps", "Valeur cible")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Prédictions vs réel
# ─────────────────────────────────────────────

def plot_predictions(test_dates, y_true: np.ndarray, preds: dict):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(test_dates, y_true.reshape(-1),
            label="Réel", color=_color("Réel"),
            linewidth=2.0, zorder=5)

    styles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    for i, (name, pred) in enumerate(preds.items()):
        if name == "Naïf":
            continue
        ax.plot(test_dates, pred.reshape(-1),
                label=name, color=_color(name),
                linestyle=styles[i % len(styles)],
                linewidth=1.3, alpha=0.85)

    _style(ax, "Prédictions vs valeurs réelles", "Temps", "Valeur cible")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Courbes d'apprentissage
# ─────────────────────────────────────────────

def plot_learning_curves(histories: dict):
    fig, ax = plt.subplots(figsize=(9, 4))

    for name, hist in histories.items():
        vals = hist.get("val_loss", [])
        if vals:
            ax.plot(vals, label=name, color=_color(name), linewidth=1.5)

    _style(ax, "Courbes d'apprentissage (validation MSE)", "Epoch", "Val MSE")
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Heatmap attention
# ─────────────────────────────────────────────

def plot_attention_heatmap(alpha: np.ndarray):
    n_rows = min(60, alpha.shape[0])
    fig, ax = plt.subplots(figsize=(7.5, 3.5))

    img = ax.imshow(alpha[-n_rows:], aspect="auto", origin="lower",
                    cmap="magma")
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Poids α_t", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Lag dans la fenêtre", fontsize=9)
    ax.set_ylabel("Observations test récentes", fontsize=9)
    _style(ax, "Heatmap des poids d'attention temporelle")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Barres de métriques
# ─────────────────────────────────────────────

def plot_metric_bars(metrics_df: pd.DataFrame, metric: str = "MAE"):
    ordered = metrics_df.sort_values(metric).copy()
    colors  = [_color(m) for m in ordered["Modèle"]]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(ordered["Modèle"], ordered[metric],
                   color=colors, height=0.55, alpha=0.85)

    for bar, val in zip(bars, ordered[metric]):
        ax.text(bar.get_width() + 0.0005 * abs(bar.get_width()) + 1e-9,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=8, color="#c9d1d9")

    _style(ax, f"Comparaison des modèles — {metric}", metric, "")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Régimes de volatilité
# ─────────────────────────────────────────────

def plot_volatility_regimes(regime_df: pd.DataFrame):
    """
    Barres groupées : MAE par modèle et par régime de volatilité.
    """
    if regime_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "Données insuffisantes", ha="center", va="center",
                transform=ax.transAxes, color="#c9d1d9")
        return fig

    regimes = ["Faible", "Normale", "Forte"]
    models  = regime_df["Modèle"].unique().tolist()

    x = np.arange(len(regimes))
    width = 0.75 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 4))

    for i, model in enumerate(models):
        sub = regime_df[regime_df["Modèle"] == model]
        maes = []
        for r in regimes:
            row = sub[sub["Régime"] == r]
            maes.append(float(row["MAE"].values[0]) if not row.empty else 0.0)

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, maes, width * 0.9,
                      label=model, color=_color(model), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    _style(ax, "MAE par régime de volatilité", "Régime", "MAE")
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Uniformité de l'attention
# ─────────────────────────────────────────────

def attention_uniformity(alpha: np.ndarray) -> tuple[float, float, float]:
    mean_val = float(np.mean(alpha))
    std_val  = float(np.std(alpha))
    cv       = std_val / mean_val if mean_val != 0 else 0.0
    return mean_val, std_val, cv


# ─────────────────────────────────────────────
# Export PDF complet
# ─────────────────────────────────────────────

def export_pdf_report(
    series_df:    pd.DataFrame,
    results:      dict,
    regime_df:    pd.DataFrame,
    metric_choice: str = "MAE",
) -> bytes:
    """
    Génère un rapport PDF complet en mémoire et retourne les bytes.
    """
    buf = io.BytesIO()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    metrics_df = results["metrics"]
    preds      = results["predictions"]
    y_true     = results["y_true"]
    test_dates = results["test_dates"]
    histories  = results["histories"]
    alpha      = results["attention"]

    with PdfPages(buf) as pdf:

        # ── Page de garde ───────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#0d1117")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("#0d1117")
        ax.axis("off")

        ax.text(0.5, 0.75, "RNN / LSTM / Attention",
                ha="center", va="center", fontsize=28,
                fontweight="bold", color="#7c3aed", transform=ax.transAxes)
        ax.text(0.5, 0.62, "Rapport d'analyse — Séries Temporelles",
                ha="center", va="center", fontsize=16,
                color="#c9d1d9", transform=ax.transAxes)
        ax.text(0.5, 0.50, f"Généré le {now}",
                ha="center", va="center", fontsize=12,
                color="#6b7280", transform=ax.transAxes)

        info_lines = [
            f"Observations : {len(series_df)}",
            f"Fenêtre : {results['window']} pas",
            f"Epochs : {results['epochs']}",
            f"Batch size : {results['batch_size']}",
            f"Seed : {results['seed']}",
        ]
        for j, line in enumerate(info_lines):
            ax.text(0.5, 0.37 - j * 0.055, line,
                    ha="center", va="center", fontsize=11,
                    color="#94a3b8", transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Série originale + transformée ───────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(series_df["date"], series_df["value"],
                     color="#3b82f6", linewidth=1.2)
        _style(axes[0], "Série originale", "Temps", "Valeur")
        axes[1].plot(series_df["date"], series_df["target"],
                     color="#10b981", linewidth=1.1)
        axes[1].axhline(0, color="#6b7280", linewidth=0.7, linestyle="--")
        _style(axes[1], "Série cible après transformation", "Temps", "Valeur cible")
        fig.suptitle("Données", fontsize=13, fontweight="bold",
                     color="#c9d1d9", y=1.01)
        fig.tight_layout()
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Tableau métriques ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 3.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")

        cols = ["Modèle", "MAE", "RMSE", "MAPE (%)", "R²"]
        table_data = []
        for _, row in metrics_df.iterrows():
            table_data.append([
                row["Modèle"],
                f"{row['MAE']:.6f}",
                f"{row['RMSE']:.6f}",
                f"{row['MAPE (%)']:.2f}",
                f"{row['R²']:.4f}",
            ])

        tbl = ax.table(
            cellText=table_data,
            colLabels=cols,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.8)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#30363d")
            if r == 0:
                cell.set_facecolor("#1f1f2e")
                cell.set_text_props(color="#c9d1d9", fontweight="bold")
            else:
                cell.set_facecolor("#0d1117" if r % 2 == 0 else "#161b22")
                cell.set_text_props(color="#c9d1d9")

        ax.set_title("Métriques comparatives", fontsize=13, fontweight="bold",
                     color="#c9d1d9", pad=14)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Métriques bar chart ──────────────────────────────────
        fig = plot_metric_bars(metrics_df, metric_choice)
        fig.suptitle(f"Comparaison — {metric_choice}", fontsize=12,
                     color="#c9d1d9", fontweight="bold")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Prédictions ──────────────────────────────────────────
        fig = plot_predictions(test_dates, y_true, preds)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Courbes d'apprentissage ──────────────────────────────
        fig = plot_learning_curves(histories)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Heatmap attention ────────────────────────────────────
        fig = plot_attention_heatmap(alpha)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Régimes de volatilité ────────────────────────────────
        if not regime_df.empty:
            fig = plot_volatility_regimes(regime_df)
            fig.suptitle("Analyse par régime de volatilité",
                         fontsize=12, color="#c9d1d9", fontweight="bold")
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

        # ── Pied de page info ────────────────────────────────────
        d = pdf.infodict()
        d["Title"]   = "RNN/LSTM/Attention — Rapport"
        d["Author"]  = "Dashboard expérimental"
        d["Subject"] = "Analyse de séries temporelles"
        d["Keywords"] = "RNN LSTM Attention Deep Learning"
        d["CreationDate"] = datetime.datetime.now()

    buf.seek(0)
    return buf.read()