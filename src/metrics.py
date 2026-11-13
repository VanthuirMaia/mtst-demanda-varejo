"""Métricas. Arrays (n_series, horizon); resultado = média sobre séries."""
import numpy as np

from src import config as C


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean(axis=1).mean()


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=1)).mean()


def smape(y_true, y_pred):
    """sMAPE em % (0-200). Pontos com y=ŷ=0 contam como erro 0."""
    den = np.abs(y_true) + np.abs(y_pred)
    err = np.where(den == 0, 0.0, 200 * np.abs(y_true - y_pred) / np.where(den == 0, 1, den))
    return err.mean(axis=1).mean()


def mase(y_true, y_pred, y_train, m=C.SEASONALITY):
    """MASE; escala = MAE do naive sazonal (lag m) no treino de cada série.
    y_train: (n_series, T_treino)."""
    escala = np.abs(y_train[:, m:] - y_train[:, :-m]).mean(axis=1)
    escala = np.where(escala == 0, np.nan, escala)
    return (np.abs(y_true - y_pred).mean(axis=1) / escala).mean()


def normalizar(y, y_train):
    """Padroniza cada série (linha) com média e desvio (ddof=1) do treino: protocolo do artigo."""
    mu = y_train.mean(axis=1, keepdims=True)
    sd = y_train.std(axis=1, ddof=1, keepdims=True)
    return (y - mu) / np.where(sd == 0, 1.0, sd)


def mse_norm(y_true, y_pred, y_train):
    return ((normalizar(y_true, y_train) - normalizar(y_pred, y_train)) ** 2).mean(axis=1).mean()


def mae_norm(y_true, y_pred, y_train):
    return np.abs(normalizar(y_true, y_train) - normalizar(y_pred, y_train)).mean(axis=1).mean()


# Duas famílias: escala original (mae, rmse, smape, mase) e normalizada (mse_norm, mae_norm)
METRICAS = ["mae", "rmse", "smape", "mase", "mse_norm", "mae_norm"]


def evaluate(y_true, y_pred, y_train):
    y_true, y_pred, y_train = (np.asarray(a, dtype="float64") for a in (y_true, y_pred, y_train))
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train),
        "mse_norm": mse_norm(y_true, y_pred, y_train),
        "mae_norm": mae_norm(y_true, y_pred, y_train),
    }
