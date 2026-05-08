# RNN · LSTM · Attention — Dashboard expérimental

Application Streamlit pour comparer des architectures récurrentes sur n'importe quelle série temporelle CSV.

## Modèles disponibles

| Modèle | Paramètres approx. |
|--------|-------------------|
| RNN vanilla | ~2 000 |
| GRU | ~25 000 |
| LSTM Baseline | ~29 000 |
| LSTM + Attention temporelle | ~30 000 |
| Transformer lite | ~15 000 |

## Fonctionnalités

- **Import CSV universel** : toute série numérique (finance, ventes, températures…)
- **Transformations** : log-rendement, différence, aucune
- **Sélection des modèles** : entraîne uniquement ce que tu veux
- **Métriques** : MAE, RMSE, MAPE, R²
- **Visualisations** : prédictions vs réel, courbes d'apprentissage, heatmap d'attention
- **Analyse par régime de volatilité** : MAE segmenté Faible / Normale / Forte
- **Export PDF complet** : rapport auto-généré

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
├── app.py           # Dashboard principal
├── models.py        # RNN, GRU, LSTM, LSTM+Att, Transformer
├── data_utils.py    # Préparation, fenêtres, régimes
├── train_utils.py   # Pipeline d'entraînement
├── plot_utils.py    # Graphiques + export PDF
└── requirements.txt
```

## Notes méthodologiques

- Le scaler est ajusté **uniquement sur le train** (pas de data leakage)
- Split chronologique : 70% train / 15% val / 15% test
- Early stopping (patience=8) + ReduceLROnPlateau
- Les poids d'attention α_t mesurent l'**importance statistique**, pas la causalité économique