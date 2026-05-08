import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import streamlit as st

from data_utils import prepare_series, detect_date_columns, compute_volatility_regimes
from train_utils import train_models
from plot_utils import (
    plot_original_series,
    plot_prepared_series,
    plot_predictions,
    plot_learning_curves,
    plot_attention_heatmap,
    plot_metric_bars,
    plot_volatility_regimes,
    attention_uniformity,
    export_pdf_report,
)


# ═══════════════════════════════════════════════════════════════
# CONFIG PAGE
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RNN · LSTM · Attention — Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# CSS GLOBAL
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0a1e 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7c3aed;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Titres principaux ───────────────────── */
.main-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(100deg, #7c3aed 20%, #06b6d4 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem;
}
.subtitle {
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 1.6rem;
}

/* ── Cartes hero / info ─────────────────── */
.hero-card {
    padding: 1.2rem 1.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg,
        rgba(124,58,237,0.15) 0%,
        rgba(6,182,212,0.08) 100%);
    border: 1px solid rgba(124,58,237,0.3);
    margin-bottom: 1.4rem;
    font-size: 0.96rem;
    color: #c9d1d9;
    line-height: 1.6;
}

/* ── Badges de statut ───────────────────── */
.badge-purple {
    display: inline-block;
    padding: 0.18rem 0.7rem;
    border-radius: 99px;
    background: rgba(124,58,237,0.25);
    color: #a78bfa;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.badge-teal {
    display: inline-block;
    padding: 0.18rem 0.7rem;
    border-radius: 99px;
    background: rgba(6,182,212,0.20);
    color: #67e8f9;
    font-size: 0.82rem;
    font-weight: 600;
}

/* ── Bouton principal ───────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #7c3aed, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.50) !important;
}

/* ── Séparateur stylé ───────────────────── */
.sep { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }

/* ── Tableau métriques ──────────────────── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Onglets ────────────────────────────── */
[data-testid="stTabs"] button {
    font-weight: 600 !important;
}

/* ── Marge padding générale ─────────────── */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2rem !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.markdown("## 🧠")
with col_title:
    st.markdown('<div class="main-title">RNN · LSTM · Attention</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dashboard comparatif de modèles séquentiels sur séries temporelles</div>',
                unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
<b>Comment ça marche ?</b><br>
Importez un CSV, choisissez la colonne à prédire, configurez les hyperparamètres
et comparez jusqu'à <b>5 architectures</b> : RNN vanilla, GRU, LSTM, LSTM+Attention et Transformer lite.
Le rapport PDF est généré automatiquement à la fin.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📂 Données")
    uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

    st.markdown("---")
    st.markdown("## ⚙️ Hyperparamètres")

    window = st.slider("Fenêtre temporelle (w)", 5, 120, 30, step=5,
                       help="Nombre de pas de temps passés utilisés comme entrée.")
    epochs = st.slider("Epochs max", 5, 150, 25, step=5,
                       help="L'early stopping arrête avant si la validation ne s'améliore plus.")
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    seed = st.number_input("Seed", 0, 9999, 42, step=1)

    st.markdown("---")
    st.markdown("## 🤖 Modèles à entraîner")

    all_models = [
        "RNN vanilla",
        "GRU",
        "LSTM Baseline",
        "LSTM + Attention temporelle",
        "Transformer lite",
    ]

    selected_models = []
    for m in all_models:
        default = m in ["GRU", "LSTM Baseline", "LSTM + Attention temporelle"]
        if st.checkbox(m, value=default):
            selected_models.append(m)

    if not selected_models:
        st.warning("Sélectionne au moins un modèle.")

    st.markdown("---")
    st.caption(
        "💡 Conseil : commence avec 10–15 epochs pour un test rapide. "
        "L'early stopping (patience=8) gère l'arrêt optimal."
    )


# ═══════════════════════════════════════════════════════════════
# LANDING (pas de CSV)
# ═══════════════════════════════════════════════════════════════

if uploaded is None:
    st.info("👈 Importe un fichier CSV dans la barre latérale pour commencer.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Modèles disponibles", "5")
    c2.metric("Métriques", "MAE · RMSE · MAPE · R²")
    c3.metric("Export", "PDF complet")
    c4.metric("Régimes", "Vol. Faible/Normale/Forte")
    c5.metric("Attention", "Heatmap α_t")

    st.markdown("### Format CSV conseillé")
    st.dataframe(pd.DataFrame({
        "Date":  ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Close": [3250.52, 3271.11, 3234.85],
    }), use_container_width=True)

    st.markdown("""
    Compatible avec toute série numérique : prix, ventes, températures,
    trafic, consommation, indices financiers, etc.
    """)
    st.stop()


# ═══════════════════════════════════════════════════════════════
# LECTURE CSV
# ═══════════════════════════════════════════════════════════════

try:
    df = pd.read_csv(uploaded)
    if df.empty:
        st.error("Le fichier CSV est vide.")
        st.stop()
except Exception as e:
    st.error(f"Impossible de lire le CSV : {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# ONGLETS
# ═══════════════════════════════════════════════════════════════

tab_data, tab_train, tab_results, tab_interp, tab_vol = st.tabs([
    "📁 Données",
    "⚙️ Entraînement",
    "📊 Résultats",
    "🧠 Interprétation",
    "📉 Volatilité",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — DONNÉES
# ═══════════════════════════════════════════════════════════════

with tab_data:
    st.subheader("Aperçu du fichier importé")

    with st.expander("Voir les données brutes", expanded=False):
        st.dataframe(df, use_container_width=True)

    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    detected_dates = detect_date_columns(df)
    date_options  = ["Aucune"] + df.columns.tolist()
    default_date  = (date_options.index(detected_dates[0])
                     if detected_dates and detected_dates[0] in date_options else 0)

    if not numeric_cols:
        st.error("Aucune colonne numérique détectée.")
        st.stop()

    st.markdown("### Configuration de la série")
    c1, c2, c3 = st.columns(3)

    with c1:
        value_col = st.selectbox("Colonne à prédire", numeric_cols)
    with c2:
        date_col = st.selectbox("Colonne date", date_options, index=default_date)
    with c3:
        transform = st.selectbox("Transformation",
                                 ["Log-rendement", "Différence", "Aucune"], index=0)

    try:
        series_df = prepare_series(df, value_col, date_col, transform)
    except ValueError as e:
        st.error(f"Erreur de préparation : {e}")
        st.stop()

    st.markdown("### Statistiques de la série préparée")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Observations", len(series_df))
    m2.metric("Moyenne", f"{series_df['target'].mean():.5f}")
    m3.metric("Écart-type", f"{series_df['target'].std():.5f}")
    m4.metric("Min", f"{series_df['target'].min():.5f}")
    m5.metric("Max", f"{series_df['target'].max():.5f}")

    if int(series_df["target"].isna().sum()) > 0:
        st.warning(f"⚠️ {int(series_df['target'].isna().sum())} valeurs manquantes détectées et supprimées.")

    if len(series_df) < window + 50:
        st.warning(
            f"La série ({len(series_df)} pts) est courte pour la fenêtre w={window}. "
            "Résultats potentiellement instables."
        )

    c_left, c_right = st.columns(2)
    with c_left:
        st.pyplot(plot_original_series(series_df), use_container_width=True)
    with c_right:
        st.pyplot(plot_prepared_series(series_df), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

with tab_train:
    st.subheader("Lancer l'entraînement")

    if not selected_models:
        st.error("Aucun modèle sélectionné dans la barre latérale.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fenêtre", f"{window} pas")
    c2.metric("Epochs max", epochs)
    c3.metric("Batch size", batch_size)
    c4.metric("Seed", seed)

    st.markdown("**Modèles sélectionnés :**")
    badge_html = "".join(
        f'<span class="badge-purple">{m}</span>' for m in selected_models
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    start = st.button(
        "🚀 Lancer l'entraînement et comparer les modèles",
        type="primary",
        disabled=not selected_models,
    )

    if start:
        progress_bar = st.progress(0)
        status_box   = st.empty()

        def _progress(val: float, msg: str):
            progress_bar.progress(min(val, 1.0))
            status_box.info(f"⏳ {msg}")

        try:
            with st.spinner("Entraînement en cours…"):
                results = train_models(
                    series_df=series_df,
                    window=window,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=seed,
                    selected_models=selected_models,
                    progress_callback=_progress,
                )

            st.session_state["results"]   = results
            st.session_state["series_df"] = series_df

            progress_bar.progress(1.0)
            status_box.success("✅ Entraînement terminé — consultez les onglets Résultats, Interprétation et Volatilité.")

        except ValueError as e:
            st.error(f"Erreur de données : {e}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")
            st.exception(e)

    if "results" not in st.session_state:
        st.info("Lance l'entraînement pour générer les résultats.")


# ═══════════════════════════════════════════════════════════════
# ONGLETS RÉSULTATS — disponibles après entraînement
# ═══════════════════════════════════════════════════════════════

if "results" in st.session_state:
    results   = st.session_state["results"]
    series_df = st.session_state["series_df"]


    # ─────────────────────────────────────────────
    # TAB 3 — RÉSULTATS
    # ─────────────────────────────────────────────

    with tab_results:
        metrics_df = results["metrics"].copy()

        st.subheader("Tableau des métriques")
        st.dataframe(
            metrics_df.style.format({
                "MAE":      "{:.6f}",
                "RMSE":     "{:.6f}",
                "MAPE (%)": "{:.2f}",
                "R²":       "{:.4f}",
            }).highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#1a3a2a")
              .highlight_max(subset=["R²"],                        color="#1a3a2a"),
            use_container_width=True,
        )

        # KPIs
        best_mae  = metrics_df.sort_values("MAE").iloc[0]
        best_rmse = metrics_df.sort_values("RMSE").iloc[0]
        best_r2   = metrics_df.sort_values("R²", ascending=False).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Meilleur MAE",  best_mae["Modèle"],  f"{best_mae['MAE']:.6f}")
        c2.metric("🏆 Meilleur RMSE", best_rmse["Modèle"], f"{best_rmse['RMSE']:.6f}")
        c3.metric("🏆 Meilleur R²",   best_r2["Modèle"],   f"{best_r2['R²']:.4f}")
        c4.metric("📐 Observations test", results["n_test"])

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # Bar chart métrique
        metric_choice = st.selectbox(
            "Métrique à visualiser", ["MAE", "RMSE", "MAPE (%)", "R²"], key="metric_sel"
        )
        _, col_fig, _ = st.columns([0.5, 3, 0.5])
        with col_fig:
            st.pyplot(plot_metric_bars(metrics_df, metric_choice),
                      use_container_width=True)

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # Prédictions vs réel
        st.markdown("### Prédictions vs valeurs réelles")
        st.pyplot(plot_predictions(
            test_dates=results["test_dates"],
            y_true=results["y_true"],
            preds=results["predictions"],
        ), use_container_width=True)

        # Courbes d'apprentissage
        st.markdown("### Courbes d'apprentissage")
        st.pyplot(plot_learning_curves(results["histories"]),
                  use_container_width=True)

        # Download métriques CSV
        st.download_button(
            "⬇️ Télécharger les métriques (CSV)",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="metrics_comparison.csv",
            mime="text/csv",
        )


    # ─────────────────────────────────────────────
    # TAB 4 — INTERPRÉTATION
    # ─────────────────────────────────────────────

    with tab_interp:
        alpha = results["attention"]
        mean_alpha, std_alpha, cv_alpha = attention_uniformity(alpha)

        st.subheader("Heatmap des poids d'attention temporelle")

        c1, c2, c3 = st.columns(3)
        c1.metric("Moyenne α_t", f"{mean_alpha:.5f}")
        c2.metric("Écart-type",   f"{std_alpha:.5f}")
        c3.metric("Coefficient variation", f"{cv_alpha:.4f}",
                  help="< 0.15 : attention quasi-uniforme | ≥ 0.15 : sélection marquée")

        _, col_hm, _ = st.columns([0.5, 3, 0.5])
        with col_hm:
            st.pyplot(plot_attention_heatmap(alpha), use_container_width=True)

        if cv_alpha < 0.15:
            st.warning(
                "⚠️ Les poids d'attention sont quasi-uniformes. "
                "La couche ne discrimine pas fortement certains instants passés — "
                "cela peut s'expliquer par la nature bruitée des rendements "
                "ou la taille de fenêtre choisie."
            )
        else:
            st.success(
                "✅ Les poids montrent une variabilité notable. "
                "Le modèle accorde plus d'importance à certains instants passés spécifiques."
            )

        st.markdown('<hr class="sep">', unsafe_allow_html=True)
        st.markdown("### Conclusion automatique")

        metrics_df = results["metrics"]

        lstm_row = metrics_df[metrics_df["Modèle"] == "LSTM Baseline"]
        att_row  = metrics_df[metrics_df["Modèle"] == "LSTM + Attention temporelle"]

        if not lstm_row.empty and not att_row.empty:
            diff_mae = att_row.iloc[0]["MAE"] - lstm_row.iloc[0]["MAE"]
            if diff_mae < 0:
                st.success(
                    f"✅ LSTM + Attention obtient un MAE inférieur au LSTM baseline "
                    f"(Δ = {diff_mae:.6f}). Ce résultat doit être confirmé sur plusieurs seeds."
                )
            else:
                st.info(
                    f"ℹ️ Le LSTM baseline reste meilleur selon le MAE "
                    f"(Δ = +{diff_mae:.6f} pour l'attention). "
                    "L'attention apporte ici surtout une lisibilité des instants passés influents."
                )
        else:
            st.info("Lance les deux modèles (LSTM Baseline + LSTM + Attention) pour la comparaison directe.")

        with st.expander("📖 Interprétation scientifique", expanded=False):
            st.markdown("""
**Que mesurent les poids α_t ?**

Les poids d'attention indiquent l'**importance statistique** qu'accorde le modèle à chaque
instant passé de la fenêtre — ils ne prouvent pas de causalité économique directe.

**Patterns typiques observés**
- **Pic à j−5** : possible effet hebdomadaire (5 jours ouvrés)
- **Pic à j−21** : possible effet mensuel (~21 jours ouvrés)
- **Distribution uniforme** : données trop bruitées pour discriminer

**Limites**
- Un seul actif analysé : généralisation à confirmer
- Données univariées : l'ajout de volume, RSI, MACD pourrait enrichir le signal
- L'attention n'est pas automatiquement supérieure : son apport dépend
  de la qualité des données, de la longueur de fenêtre et des hyperparamètres
""")

    # ─────────────────────────────────────────────
    # TAB 5 — VOLATILITÉ
    # ─────────────────────────────────────────────

    with tab_vol:
        st.subheader("Analyse par régime de volatilité")

        st.markdown("""
La série est divisée en **3 régimes** selon la volatilité réalisée glissante :
- 🟢 **Faible** (33e percentile)
- 🟡 **Normale** (33e – 67e percentile)
- 🔴 **Forte** (au-dessus du 67e percentile)

Le MAE de chaque modèle est calculé séparément dans chaque régime.
        """)

        window_vol = st.slider("Fenêtre volatilité (jours)", 5, 60, 20, step=5)

        try:
            regime_df = compute_volatility_regimes(
                series_df=series_df,
                y_true=results["y_true"],
                preds=results["predictions"],
                test_start_index=results["test_start_index"],
                window_vol=window_vol,
            )
        except Exception as e:
            st.error(f"Erreur régime volatilité : {e}")
            regime_df = pd.DataFrame()

        if not regime_df.empty:
            st.dataframe(
                regime_df.pivot_table(
                    index="Modèle", columns="Régime", values="MAE"
                ).round(6),
                use_container_width=True,
            )

            st.pyplot(plot_volatility_regimes(regime_df), use_container_width=True)

            # Insight automatique
            if "LSTM + Attention temporelle" in regime_df["Modèle"].values:
                att_forte = regime_df[
                    (regime_df["Modèle"] == "LSTM + Attention temporelle") &
                    (regime_df["Régime"] == "Forte")
                ]
                lstm_forte = regime_df[
                    (regime_df["Modèle"] == "LSTM Baseline") &
                    (regime_df["Régime"] == "Forte")
                ]
                if not att_forte.empty and not lstm_forte.empty:
                    if att_forte.iloc[0]["MAE"] < lstm_forte.iloc[0]["MAE"]:
                        st.success(
                            "✅ En régime de **forte volatilité**, l'attention temporelle surpasse "
                            "le LSTM baseline — son mécanisme de sélection temporelle est ici avantageux."
                        )
                    else:
                        st.info(
                            "ℹ️ En régime de **forte volatilité**, le LSTM baseline reste meilleur. "
                            "La volatilité élevée rend la sélection temporelle plus difficile."
                        )
        else:
            st.warning("Données insuffisantes pour l'analyse par régime.")

    # ─────────────────────────────────────────────
    # EXPORT PDF — disponible dans tous les onglets
    # ─────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 📄 Exporter le rapport complet")

    col_pdf, col_hint = st.columns([1, 3])

    with col_pdf:
        if st.button("🖨️ Générer le rapport PDF", type="primary"):
            with st.spinner("Génération du PDF en cours…"):
                try:
                    regime_df_export = compute_volatility_regimes(
                        series_df=series_df,
                        y_true=results["y_true"],
                        preds=results["predictions"],
                        test_start_index=results["test_start_index"],
                        window_vol=20,
                    )
                except Exception:
                    regime_df_export = pd.DataFrame()

                pdf_bytes = export_pdf_report(
                    series_df=series_df,
                    results=results,
                    regime_df=regime_df_export,
                    metric_choice=st.session_state.get("metric_sel", "MAE"),
                )

            st.download_button(
                "⬇️ Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name="rapport_rnn_lstm_attention.pdf",
                mime="application/pdf",
            )

    with col_hint:
        st.markdown(
            '<span class="badge-teal">PDF</span> '
            "Le rapport inclut : page de garde · données · métriques · "
            "graphes prédictions · courbes d'apprentissage · heatmap d'attention · analyse régimes.",
            unsafe_allow_html=True,
        )

else:
    for tab in [tab_results, tab_interp, tab_vol]:
        with tab:
            st.info("⚙️ Lance l'entraînement dans l'onglet **Entraînement** pour voir les résultats ici.")