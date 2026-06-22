# 🧠 RNN · LSTM · Attention — Time Series Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?logo=streamlit)
![Models](https://img.shields.io/badge/Models-5%20architectures-purple)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive Streamlit dashboard to **train, compare and interpret** sequential deep learning architectures on any time series CSV — with attention heatmaps, volatility regime analysis and automatic PDF report generation.

---

## 📸 Demo

> _Add screenshot after deployment: `![Demo](assets/demo.png)`_

---

## 🤖 Available Models

| Model | Architecture | Parameters |
|-------|-------------|------------|
| RNN vanilla | SimpleRNN → Dense | ~2,000 |
| GRU | GRU × 2 → Dense | ~25,000 |
| LSTM Baseline | LSTM × 2 → Dense | ~29,000 |
| **LSTM + Temporal Attention** | LSTM × 2 → TemporalAttention → Dense | ~30,000 |
| Transformer lite | MultiHeadAttention + FFN → Dense | ~15,000 |

---

## ✨ Features

- 📂 **Universal CSV import** — any numeric time series (finance, sales, temperature, traffic…)
- 🔄 **Transformations** — log-return, difference, or raw values
- ⚙️ **Configurable hyperparameters** — window size, epochs, batch size, seed
- 🧠 **Custom Temporal Attention layer** — built from scratch in TensorFlow/Keras
- 📊 **4 metrics** — MAE, RMSE, MAPE, R²
- 🔍 **Attention heatmap** — visualize which past timesteps the model focuses on
- 📉 **Volatility regime analysis** — MAE segmented by Low / Normal / High volatility
- 📄 **Auto PDF report** — full report generated with one click
- 🛡️ **No data leakage** — scaler fitted only on train set, chronological split (70/15/15)

---

## 🧠 Custom Attention Mechanism

```python
# Temporal Attention (implemented from scratch)
e_t   = v^T · tanh(W · h_t)   # score per timestep
alpha = softmax(e_t)            # attention weights
c     = Σ alpha_t · h_t        # context vector
```

The attention weights α_t measure the **statistical importance** of each past timestep — not economic causality.

---

## 🚀 Run Locally

```bash
git clone https://github.com/wll-hayat04/rnn-lstm-attention-dashboard.git
cd rnn-lstm-attention-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
rnn-lstm-attention-dashboard/
│
├── app.py           # Main Streamlit dashboard
├── models.py        # RNN, GRU, LSTM, LSTM+Attention, Transformer
├── data_utils.py    # Series preparation, windows, volatility regimes
├── train_utils.py   # Training pipeline + metrics
├── plot_utils.py    # Charts + PDF export
├── requirements.txt
└── .python-version  # Python 3.11
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| TensorFlow 2.17 (CPU) | Deep learning models |
| Streamlit | Interactive dashboard |
| scikit-learn | Preprocessing + metrics |
| pandas / numpy | Data manipulation |
| matplotlib | Visualizations |
| reportlab | PDF generation |

---

## 📊 Methodology

- Chronological split: **70% train / 15% val / 15% test**
- Early stopping (patience=8) + ReduceLROnPlateau
- Scaler fitted **only on train** set (no data leakage)
- Attention weights measure statistical importance, not causality

---

## 👩‍💻 Author

**Hayat** — 4th Year Engineering Student  
🌍 Morocco | 💼 Open to freelance & internships  
[![GitHub](https://img.shields.io/badge/GitHub-wll--hayat04-181717?logo=github)](https://github.com/wll-hayat04)

---

## 📄 License

[MIT License](LICENSE)
