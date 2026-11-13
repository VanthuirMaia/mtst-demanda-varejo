"""Figuras de comparação (baselines + modelos globais) do dataset C.DATASET; a tabela está em src/final.py.
Só entra o que existir em results/tables e data/processed (modelos ainda não treinados são ignorados)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.data import get_test_origins, load_data

P = C.PREFIX
COR = {"real": "#333333", "naive_anual": "#8a8a8a", "arima": "#6a8f3a",
       "mtst_global": "#2a6fb0", "patchtst_global": "#d9822b"}
NOME = {"real": "real", "naive_anual": "naive anual", "arima": "ARIMA", "mtst_global": "MTST global",
        "patchtst_global": "PatchTST global"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def fig_previsoes(mat):
    vals = mat.to_numpy(dtype="float64")
    origem = get_test_origins()[0]
    o = mat.index.get_loc(origem)
    prev = {}
    for nome in ["arima", "mtst_global", "patchtst_global"]:
        f = C.PROCESSED_DIR / f"pred_{P}{nome}.npy"
        if f.exists():
            prev[nome] = np.load(f)[0]  # janela 1

    # séries de escala baixa, média e alta (quantis da média no treino)
    escala = mat.loc[:C.TRAIN_END].mean().sort_values()
    idx = [escala.index[int(q * (len(escala) - 1))] for q in (0.03, 0.35, 0.65, 0.97)]
    ctx = 28
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=False)
    for ax, s in zip(axs.ravel(), idx):
        j = mat.columns.get_loc(s)
        datas = mat.index[o - ctx:o + C.HORIZON]
        ax.plot(datas, vals[o - ctx:o + C.HORIZON, j], color=COR["real"], lw=1.6, label=NOME["real"])
        fut = mat.index[o:o + C.HORIZON]
        for nome, p in prev.items():
            ax.plot(fut, p[j], color=COR[nome], lw=1.2, label=NOME[nome])
        ax.axvline(origem, color="#aaaaaa", lw=1)
        ax.set_title(f"{s} (média no treino: {escala[s]:.0f})", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"Janela 1 de teste (origem {origem.date()})", y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(C.FIG_DIR / f"{P}previsoes_exemplo.png")
    plt.close(fig)


def fig_curvas():
    fig, ax = plt.subplots(figsize=(6, 4))
    for nome in ["mtst_global", "patchtst_global"]:
        f = C.TAB_DIR / f"{P}{nome}_log.csv"
        if not f.exists():
            continue
        log = pd.read_csv(f)
        ax.plot(log.epoca, log.loss_treino, color=COR[nome], marker="o", ms=3, label=f"{NOME[nome]} treino")
        ax.plot(log.epoca, log.loss_val, color=COR[nome], marker="o", ms=3, ls="--", label=f"{NOME[nome]} val")
    ax.set(xlabel="época", ylabel="MSE (normalizado)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"{P}curvas_treino.png")
    plt.close(fig)


if __name__ == "__main__":
    mat = load_data()
    fig_previsoes(mat)
    fig_curvas()
