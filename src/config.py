"""Configuração central do projeto."""
from pathlib import Path

import pandas as pd

# --- Caminhos ---
RAIZ = Path(__file__).resolve().parent.parent
RAW_DIR = RAIZ / "data" / "raw"
PROCESSED_DIR = RAIZ / "data" / "processed"
TRAIN_CSV = RAW_DIR / "train.csv"
CKPT_DIR = RAIZ / "results" / "checkpoints"
FIG_DIR = RAIZ / "results" / "figures"
TAB_DIR = RAIZ / "results" / "tables"
LOG_DIR = RAIZ / "results" / "logs"

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


# --- MTST (argumentos esperados pelo construtor oficial, via SimpleNamespace) ---
# 3 ramos com patch = stride = 7, 14, 28 dias -> 52, 26, 13 tokens (lookback 364).
# O código oficial usa padding_patch='end' (acrescenta 1 patch); aqui None para ter
# exatamente 364/patch tokens.
MTST = dict(
    seq_len=LOOKBACK, pred_len=HORIZON, enc_in=1,
    e_layers=2, n_branches=3, n_heads=4, d_model=64, d_ff=128,
    dropout=0.2, fc_dropout=0.2, head_dropout=0.0,
    patch_len_ls="7, 14, 28", stride_ls="7, 14, 28", padding_patch=None,
    revin=1, affine=0, subtract_last=0, rel_pe="rel_sin",
    individual=0, decomposition=0, kernel_size=25,
    store_attn=False, pre_norm=False, res_attn=False,
)

# --- Treino ---
TREINO = dict(max_lr=5e-4, pct_start=0.3, batch=128, epocas=15,
              stride_treino=7, stride_val=7)  # Adam + OneCycleLR; sem early stopping (v2)


def to_ts(s):
    return pd.Timestamp(s)


# =============================================================================
# ETTh1 (Fase 3: replicação do experimento do artigo). Independente do varejo acima.
# Fonte: ../MTST/scripts/MTST/etth1.sh (commit d52b6b5) + defaults de ../MTST/run_longExp.py
# =============================================================================
ETTH1_CSV = RAW_DIR / "ETTh1.csv"
HORIZONTES_ETTH1 = (96, 192, 336, 720)

ETTH1 = dict(
    # modelo (etth1.sh); args não passados no .sh usam o default do run_longExp.py
    modelo=dict(
        seq_len=336, enc_in=7, e_layers=2, n_branches=2, n_heads=16, d_model=128, d_ff=256,
        dropout=0.3, fc_dropout=0.3, head_dropout=0.1,
        patch_len_ls="8, 16", stride_ls="4, 8", padding_patch="end", rel_pe="rel_sin",
        # defaults do run_longExp.py
        revin=1, affine=0, subtract_last=0, decomposition=0, kernel_size=25, individual=0,
        store_attn=False, pre_norm=False, res_attn=False, pe="zeros", no_learn_pe=True,
    ),
    # treino (etth1.sh + defaults)
    treino=dict(
        seed=2021, features="M", batch_size=256, learning_rate=1e-4, train_epochs=100,
        patience=20, lradj="type3", pct_start=0.3, l2=0.0, loss="mse",
        label_len=0,
    ),
)
