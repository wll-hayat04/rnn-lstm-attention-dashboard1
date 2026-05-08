import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, SimpleRNN, LSTM, GRU, Dense, Dropout,
    Layer, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Add
)
from tensorflow.keras import backend as K
import tensorflow as tf


# ─────────────────────────────────────────────
# Couche d'attention temporelle personnalisée
# ─────────────────────────────────────────────

class TemporalAttention(Layer):
    """
    Attention temporelle simplifiée :
        e_t   = v^T tanh(W h_t)
        alpha = softmax(e_t)
        c     = Σ alpha_t * h_t
    """

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(shape=(d, d), initializer="glorot_uniform",
                                 trainable=True, name="W_attention")
        self.v = self.add_weight(shape=(d, 1), initializer="glorot_uniform",
                                 trainable=True, name="v_attention")
        super().build(input_shape)

    def call(self, h):
        score = K.dot(K.tanh(K.dot(h, self.W)), self.v)   # (B, T, 1)
        alpha = K.softmax(score, axis=1)                    # (B, T, 1)
        context = K.sum(alpha * h, axis=1)                  # (B, d)
        return context, alpha

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────
# RNN vanilla
# ─────────────────────────────────────────────

def build_rnn_model(window: int) -> Sequential:
    model = Sequential([
        SimpleRNN(32, input_shape=(window, 1)),
        Dropout(0.2),
        Dense(1)
    ], name="RNN_vanilla")
    model.compile(optimizer="adam", loss="mse")
    return model


# ─────────────────────────────────────────────
# GRU
# ─────────────────────────────────────────────

def build_gru_model(window: int) -> Sequential:
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ], name="GRU")
    model.compile(optimizer="adam", loss="mse")
    return model


# ─────────────────────────────────────────────
# LSTM baseline
# ─────────────────────────────────────────────

def build_lstm_model(window: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ], name="LSTM_Baseline")
    model.compile(optimizer="adam", loss="mse")
    return model


# ─────────────────────────────────────────────
# LSTM + Attention temporelle
# ─────────────────────────────────────────────

def build_lstm_attention_model(window: int):
    inputs = Input(shape=(window, 1))

    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    h = LSTM(32, return_sequences=True)(x)

    context, alpha = TemporalAttention()(h)
    outputs = Dense(1)(context)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Attention")
    model.compile(optimizer="adam", loss="mse")

    attention_model = Model(inputs=inputs, outputs=alpha,
                            name="LSTM_Attention_weights")

    return model, attention_model


# ─────────────────────────────────────────────
# Transformer lite
# ─────────────────────────────────────────────

def build_transformer_model(window: int, d_model: int = 32,
                             num_heads: int = 2, ff_dim: int = 64) -> Model:
    """
    Transformer léger pour séries temporelles 1-D.
    Architecture : projection → self-attention → FFN → pooling → Dense(1)
    """
    inputs = Input(shape=(window, 1))

    # Projection vers d_model
    x = Dense(d_model)(inputs)

    # Bloc self-attention + résidu + norm
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))

    # FFN + résidu + norm
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(d_model)(ffn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))

    # Pooling temporel + sortie
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name="Transformer_lite")
    model.compile(optimizer="adam", loss="mse")
    return model