"""Configuração central do projeto."""
import os
from pathlib import Path

import pandas as pd

# --- Dataset de aplicação: "m5" (principal) | "store_item" (experimento preliminar) ---
# Pode ser sobrescrito pela variável de ambiente DATASET (ex.: DATASET=store_item).
DATASET = os.environ.get("DATASET", "m5")
assert DATASET in ("m5", "store_item"), f"DATASET inválido: {DATASET}"

# --- Caminhos ---
RAIZ = Path(__file__).resolve().parent.parent
RAW_DIR = RAIZ / "data" / "raw"
PROCESSED_DIR = RAIZ / "data" / "processed"
TRAIN_CSV = RAW_DIR / "train.csv"  # Store Item Demand Forecasting
M5_DIR = RAW_DIR / "m5"
M5_SALES_CSV = M5_DIR / "sales_train_evaluation.csv"
M5_CALENDAR_CSV = M5_DIR / "calendar.csv"
CKPT_DIR = RAIZ / "results" / "checkpoints"
TAB_BASE = RAIZ / "results" / "tables"  # M5 (prefixo m5_) e ETTh1 (etth1_)
FIG_DIR = RAIZ / "results" / "figures"
TAB_DIR = TAB_BASE
LOG_DIR = RAIZ / "results" / "logs"

# --- Reprodutibilidade ---
SEED = 42

# --- Janelas (iguais nos dois datasets) ---
# 364 dias = 52 semanas: o modelo vê o mesmo dia da semana do ano anterior (justo vs. naive
# anual) e 364 é divisível por 7, 14 e 28 (patches alinhados a semanas). O 336 do paper é de
# datasets horários e não se aplica a dados diários.
LOOKBACK = 364
HORIZON = 90
N_TEST_WINDOWS = 4  # 4 x 90 = 360 dias de teste
VAL_DAYS = 180

# --- Por dataset: prefixo dos arquivos de saída e datas ---
# Prefixo dos arquivos em results/ ("" mantém os nomes do Store Item já versionados).
PREFIX = {"m5": "m5_", "store_item": ""}[DATASET]
if DATASET == "store_item":  # experimento preliminar: results/store_item/{tables,figures,logs}
    TAB_DIR = RAIZ / "results" / "store_item" / "tables"
    FIG_DIR = RAIZ / "results" / "store_item" / "figures"
    LOG_DIR = RAIZ / "results" / "store_item" / "logs"


def usar_smoke():
    """Redireciona as saídas para results/smoke e data/processed/smoke (testes de 1 época)."""
    global TAB_DIR, FIG_DIR, PROCESSED_DIR, CKPT_DIR
    TAB_DIR = RAIZ / "results" / "smoke" / "tables"
    FIG_DIR = RAIZ / "results" / "smoke" / "figures"
    CKPT_DIR = RAIZ / "results" / "smoke" / "checkpoints"
    PROCESSED_DIR = RAIZ / "data" / "processed" / "smoke"

# M5: d_1 = 2011-01-29 e 1941 dias (sales_train_evaluation) -> último dia 2016-06-19.
M5_FIRST_DAY = "2011-01-29"
M5_N_DAYS = 1941


def calcular_splits_m5(first=M5_FIRST_DAY, n_days=M5_N_DAYS):
    """Splits a partir das datas: teste = últimos N_TEST_WINDOWS*HORIZON dias; validação =
    VAL_DAYS dias antes do teste; treino = o resto. -> dict com datas inclusivas."""
    d0 = pd.Timestamp(first)
    ultimo = d0 + pd.Timedelta(days=n_days - 1)
    test_start = ultimo - pd.Timedelta(days=N_TEST_WINDOWS * HORIZON - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=VAL_DAYS - 1)
    train_end = val_start - pd.Timedelta(days=1)
    f = lambda t: t.strftime("%Y-%m-%d")
    return dict(first=f(d0), train_end=f(train_end), val_start=f(val_start), val_end=f(val_end),
                test_start=f(test_start), last=f(ultimo))


if DATASET == "m5":
    SPLITS = calcular_splits_m5()
    TRAIN_END, VAL_END, TEST_START = SPLITS["train_end"], SPLITS["val_end"], SPLITS["test_start"]
else:
    # Store Item: 2013-01-01 a 2017-12-31. Teste: 2017 em 4 janelas de 90 dias a partir de
    # 2017-01-01 (a última termina em 2017-12-26; os 5 últimos dias não são avaliados).
    TRAIN_END = "2016-06-30"
    VAL_END = "2016-12-31"
    TEST_START = "2017-01-01"


def imprimir_splits():
    ini = pd.Timestamp(TEST_START)
    fim = ini + pd.Timedelta(days=N_TEST_WINDOWS * HORIZON - 1)
    print(f"Splits ({DATASET}): treino até {TRAIN_END} | validação até {VAL_END} | "
          f"teste {TEST_START} a {fim.date()} ({N_TEST_WINDOWS} x {HORIZON} dias)")


# --- Métricas ---
SEASONALITY = 7  # sazonalidade usada na escala do MASE

# --- Baselines ---
YEAR_LAG = 364  # 52 semanas: mesmo dia da semana um ano antes


# --- Modelos (argumentos esperados pelos construtores oficiais, via SimpleNamespace) ---
# Patches sem sobreposição (stride = patch). Lookback 364: patch 7/14/28/91 -> 52/26/13/4 tokens.
# padding_patch=None: exatamente 364/patch tokens (o oficial usa 'end', que acrescenta 1 patch).
# d_ff = 2*d_model, 4 cabeças, 2 camadas, RevIN (e RPE no MTST) ligados.
def config_modelo(modelo, patch, d_model, dropout):
    """Namespace-dict do modelo. modelo: 'mtst_global' | 'patchtst_global'.
    patch: tupla de patches (MTST) ou int (PatchTST)."""
    base = dict(seq_len=LOOKBACK, pred_len=HORIZON, enc_in=1, e_layers=2, n_heads=4,
                d_model=d_model, d_ff=2 * d_model, dropout=dropout, fc_dropout=dropout,
                head_dropout=0.0, padding_patch=None, revin=1, affine=0, subtract_last=0,
                individual=0, decomposition=0, kernel_size=25,
                store_attn=False, pre_norm=False, res_attn=False)
    if modelo == "mtst_global":
        ls = ", ".join(str(p) for p in patch)
        return dict(base, n_branches=len(patch), patch_len_ls=ls, stride_ls=ls, rel_pe="rel_sin")
    return dict(base, patch_len=patch, stride=patch)


# MTST tem dropout 0,8 a mais: seu ótimo (0,6) estava na borda; o do PatchTST (0,4) é interior.
# --- Grades da seleção de parâmetros (src/selecao.py): só treino e validação ---
GRADE = dict(
    mtst_global=dict(patch=[(7, 14, 28), (7, 28, 91)], d_model=[32, 64], dropout=[0.2, 0.4, 0.6, 0.8]),
    patchtst_global=dict(patch=[7, 14, 28], d_model=[32, 64], dropout=[0.2, 0.4, 0.6]),
)
SEEDS_FINAL = (42, 43, 44)

# Compatibilidade (config padrão anterior do MTST)
MTST = config_modelo("mtst_global", (7, 14, 28), 64, 0.2)

# --- Treino ---
# Adam + OneCycleLR, 15 épocas fixas, sem early stopping (melhor checkpoint pela validação).
# Stride das janelas de treino: 1 no M5 (todas as origens); 7 no Store Item (preliminar).
TREINO = dict(max_lr=5e-4, pct_start=0.3, batch=128, epocas=15,
              stride_treino=1 if DATASET == "m5" else 7, stride_val=7)


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
