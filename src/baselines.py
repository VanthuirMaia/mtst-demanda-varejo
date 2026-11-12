"""Baselines ingênuos sazonais, avaliados nas 4 janelas de teste."""
import numpy as np
import pandas as pd

from src import config as C
from src.data import get_test_origins, load_data
from src.metrics import evaluate


def seasonal_naive_weekly(vals, origem, horizon=C.HORIZON):
    """Repete os últimos 7 dias antes da origem até cobrir o horizonte. -> (S, horizon)"""
    ultimos = vals[origem - 7:origem].T  # (S, 7)
    reps = int(np.ceil(horizon / 7))
    return np.tile(ultimos, (1, reps))[:, :horizon]


def seasonal_naive_yearly(vals, origem, horizon=C.HORIZON):
    """Para cada dia previsto, usa o valor de 364 dias antes (mesmo dia da semana). -> (S, horizon)"""
    idx = origem + np.arange(horizon) - C.YEAR_LAG
    return vals[idx].T


def run_baselines():
    mat = load_data()
    vals = mat.to_numpy(dtype="float64")
    n_treino = mat.index.searchsorted(C.to_ts(C.TRAIN_END), side="right")
    y_train = vals[:n_treino].T  # escala do MASE: período de treino

    modelos = {"naive_semanal": seasonal_naive_weekly, "naive_anual": seasonal_naive_yearly}
    linhas = []
    for nome, f in modelos.items():
        for k, origem_data in enumerate(get_test_origins(), start=1):
            o = mat.index.get_loc(origem_data)
            y_true = vals[o:o + C.HORIZON].T
            res = evaluate(y_true, f(vals, o), y_train)
            linhas.append({"modelo": nome, "janela": k, "origem": origem_data.date(), **res})
    tab = pd.DataFrame(linhas)

    medias = tab.groupby("modelo", sort=False)[["mae", "rmse", "smape", "mase"]].mean().reset_index()
    medias["janela"] = "media"
    tab = pd.concat([tab, medias], ignore_index=True)

    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(C.TAB_DIR / "baselines.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return tab


if __name__ == "__main__":
    run_baselines()
