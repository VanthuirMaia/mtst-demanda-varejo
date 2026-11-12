"""Configuração central do projeto."""
from pathlib import Path

import pandas as pd

# --- Caminhos ---
RAIZ = Path(__file__).resolve().parent.parent
RAW_DIR = RAIZ / "data" / "raw"
PROCESSED_DIR = RAIZ / "data" / "processed"
TRAIN_CSV = RAW_DIR / "train.csv"
FIG_DIR = RAIZ / "results" / "figures"
TAB_DIR = RAIZ / "results" / "tables"

# --- Reprodutibilidade ---
SEED = 42

# --- Janelas ---
# 364 dias = 52 semanas: o modelo vê o mesmo dia da semana do ano anterior (justo vs. naive
# anual) e 364 é divisível por 7, 14 e 28 (patches alinhados a semanas). O 336 do paper é de
# datasets horários e não se aplica a dados diários.
LOOKBACK = 364
HORIZON = 90

# --- Splits (datas inclusivas) ---
TRAIN_END = "2016-06-30"
VAL_END = "2016-12-31"

# Teste: 2017 em 4 janelas de 90 dias, sem sobreposição, a partir de 2017-01-01.
# 4 x 90 = 360 dias -> a última janela termina em 2017-12-26; os 5 últimos dias
# de 2017 (27/12 a 31/12) não são avaliados.
TEST_START = "2017-01-01"
N_TEST_WINDOWS = 4

# --- Métricas ---
SEASONALITY = 7  # sazonalidade usada na escala do MASE

# --- Baselines ---
YEAR_LAG = 364  # 52 semanas: mesmo dia da semana um ano antes


def to_ts(s):
    return pd.Timestamp(s)
