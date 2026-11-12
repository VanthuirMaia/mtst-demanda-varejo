"""Carga dos dados e geração de janelas."""
import numpy as np
import pandas as pd

from src import config as C


def load_data():
    """Lê train.csv e devolve DataFrame (T dias x 500 séries), colunas 's{store}_i{item}'."""
    if not C.TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{C.TRAIN_CSV} não encontrado. Baixe train.csv do Kaggle "
            "(Store Item Demand Forecasting Challenge) e coloque em data/raw/."
        )
    df = pd.read_csv(C.TRAIN_CSV, parse_dates=["date"])
    df["serie"] = "s" + df["store"].astype(str) + "_i" + df["item"].astype(str)
    mat = df.pivot(index="date", columns="serie", values="sales")
    # ordem estável: loja, depois item
    cols = sorted(mat.columns, key=lambda c: tuple(int(p[1:]) for p in c.split("_")))
    mat = mat[cols].astype("float32")

    # Validações
    esperado = pd.date_range(mat.index.min(), mat.index.max(), freq="D")
    faltando = esperado.difference(mat.index)
    assert len(faltando) == 0, f"{len(faltando)} datas faltando, ex.: {faltando[:3].tolist()}"
    assert not mat.isna().any().any(), "Há NaN na matriz (série sem observação em alguma data)"

    print(f"Dados: shape={mat.shape} (dias x séries), "
          f"período {mat.index.min().date()} a {mat.index.max().date()}")
    return mat


def get_test_origins():
    """Datas de origem (primeiro dia previsto) das janelas de teste."""
    ini = C.to_ts(C.TEST_START)
    return [ini + pd.Timedelta(days=C.HORIZON * k) for k in range(C.N_TEST_WINDOWS)]


def make_windows(matrix, start, end, lookback=C.LOOKBACK, horizon=C.HORIZON, stride=1):
    """Gera (X, y) channel-independent: uma amostra por (série, origem).

    O horizonte previsto (y) fica inteiramente dentro de [start, end]; o lookback (X)
    pode usar dias anteriores a `start` (histórico). Origem = primeiro dia previsto.
    X: (n_amostras, lookback), y: (n_amostras, horizon). Ordem: origem (externo), série (interno).
    """
    datas = matrix.index
    vals = matrix.to_numpy(dtype="float32")  # (T, S)
    i_ini = max(datas.searchsorted(C.to_ts(start), side="left"), lookback)
    i_fim = datas.searchsorted(C.to_ts(end), side="right")  # exclusivo
    origens = np.arange(i_ini, i_fim - horizon + 1, stride)
    if len(origens) == 0:
        raise ValueError("Nenhuma janela cabe no intervalo pedido")

    idx_x = origens[:, None] + np.arange(-lookback, 0)  # (n, lookback)
    idx_y = origens[:, None] + np.arange(horizon)  # (n, horizon)
    n, S = len(origens), vals.shape[1]
    X = vals[idx_x].transpose(0, 2, 1).reshape(n * S, lookback)
    y = vals[idx_y].transpose(0, 2, 1).reshape(n * S, horizon)
    return X, y
