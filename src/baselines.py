"""Baselines clássicos avaliados nas 4 janelas de teste: naive sazonal (semanal e anual) e,
opcionalmente, ARIMA por série (AutoARIMA do statsforecast, sazonalidade 7, origem móvel)."""
import argparse
import time

import numpy as np
import pandas as pd

from src import config as C
from src.data import get_test_origins, load_data
from src.metrics import METRICAS, evaluate


def seasonal_naive_weekly(vals, origem, horizon=C.HORIZON):
    """Repete os últimos 7 dias antes da origem até cobrir o horizonte. -> (S, horizon)"""
    ultimos = vals[origem - 7:origem].T  # (S, 7)
    reps = int(np.ceil(horizon / 7))
    return np.tile(ultimos, (1, reps))[:, :horizon]


def seasonal_naive_yearly(vals, origem, horizon=C.HORIZON):
    """Para cada dia previsto, usa o valor de 364 dias antes (mesmo dia da semana). -> (S, horizon)"""
    idx = origem + np.arange(horizon) - C.YEAR_LAG
    return vals[idx].T


def arima_por_serie(mat, origem_data, horizon=C.HORIZON):
    """AutoARIMA (season_length=7) por série, ajustado em TODO o histórico antes da origem
    (treino + validação + janelas de teste anteriores: origem móvel). Previsão clipada em >= 0.
    -> (S, horizon), na ordem das colunas de `mat`."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    hist = mat.loc[:origem_data - pd.Timedelta(days=1)]
    long = (hist.rename_axis("ds").reset_index()
            .melt(id_vars="ds", var_name="unique_id", value_name="y"))
    long["y"] = long["y"].astype("float64")
    sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=-1)
    prev = sf.forecast(df=long, h=horizon)
    wide = prev.pivot(index="unique_id", columns="ds", values="AutoARIMA").loc[mat.columns]
    return np.clip(wide.to_numpy(dtype="float64"), 0, None)


def run_baselines(arima=True):
    mat = load_data()
    C.imprimir_splits()
    vals = mat.to_numpy(dtype="float64")
    n_treino = mat.index.searchsorted(C.to_ts(C.TRAIN_END), side="right")
    y_train = vals[:n_treino].T  # escala das métricas (MASE, normalização): período de treino
    origens = get_test_origins()

    previsoes = {  # nome -> lista (por janela) de (S, horizon) e tempo (s) por janela
        "naive_semanal": ([seasonal_naive_weekly(vals, mat.index.get_loc(o)) for o in origens], None),
        "naive_anual": ([seasonal_naive_yearly(vals, mat.index.get_loc(o)) for o in origens], None),
    }
    if arima:
        preds, tempos = [], []
        for k, o in enumerate(origens, start=1):
            t0 = time.time()
            preds.append(arima_por_serie(mat, o))
            tempos.append(time.time() - t0)
            print(f"ARIMA janela {k}: {tempos[-1]:.0f}s ({mat.shape[1]} séries)")
        previsoes["arima"] = (preds, tempos)
        C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(C.PROCESSED_DIR / f"pred_{C.PREFIX}arima.npy", np.stack(preds))

    linhas = []
    for nome, (preds, tempos) in previsoes.items():
        for k, (origem_data, p) in enumerate(zip(origens, preds)):
            o = mat.index.get_loc(origem_data)
            res = evaluate(vals[o:o + C.HORIZON].T, p, y_train)
            linhas.append({"modelo": nome, "janela": k + 1, "origem": origem_data.date(), **res,
                           "segundos": tempos[k] if tempos else np.nan})
    tab = pd.DataFrame(linhas)
    medias = tab.groupby("modelo", sort=False)[METRICAS].mean().reset_index()
    medias["janela"] = "media"
    medias["segundos"] = tab.groupby("modelo", sort=False)["segundos"].sum(min_count=1).values
    tab = pd.concat([tab, medias], ignore_index=True)

    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(C.TAB_DIR / f"{C.PREFIX}baselines.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-arima", action="store_true", help="só os naive (rápido)")
    run_baselines(arima=not ap.parse_args().sem_arima)
