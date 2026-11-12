"""Junta baselines + MTST global + MTST local e gera figuras de comparação."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.baselines import seasonal_naive_yearly
from src.data import get_test_origins, load_data

COR = {"real": "#333333", "naive_anual": "#8a8a8a", "mtst_global": "#2a6fb0", "mtst_local_todas": "#d9822b"}
NOME = {"real": "real", "naive_anual": "naive anual", "mtst_global": "MTST global",
        "mtst_local_todas": "MTST local (por loja)"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def tabela_comparacao():
    arqs = ["baselines.csv", "mtst_global.csv", "mtst_local_todas.csv"]
    tab = pd.concat([pd.read_csv(C.TAB_DIR / a) for a in arqs], ignore_index=True)
    tab.to_csv(C.TAB_DIR / "comparacao.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def fig_previsoes(mat):
    vals = mat.to_numpy(dtype="float64")
    o = mat.index.get_loc(get_test_origins()[0])
    prev = {"naive_anual": seasonal_naive_yearly(vals, o)}
    for nome in ["mtst_global", "mtst_local_todas"]:
        prev[nome] = np.load(C.PROCESSED_DIR / f"pred_{nome}.npy")[0]  # janela 1

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
            ax.plot(fut, p[j], color=COR[nome], lw=1.2, ls="--" if nome == "naive_anual" else "-",
                    label=NOME[nome])
        ax.axvline(mat.index[o], color="#aaaaaa", lw=1)
        ax.set_title(f"{s} (média no treino: {escala[s]:.0f})", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Janela 1 de teste (origem 2017-01-01)", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(C.FIG_DIR / "previsoes_exemplo.png")
    plt.close(fig)


def fig_curvas():
    tags = ["global", "local_loja1", "local_loja2"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, tag in zip(axs, tags):
        log = pd.read_csv(C.TAB_DIR / f"mtst_{tag}_log.csv")
        ax.plot(log.epoca, log.loss_treino, color=COR["mtst_global"], marker="o", ms=3, label="treino")
        ax.plot(log.epoca, log.loss_val, color=COR["mtst_local_todas"], marker="o", ms=3, label="validação")
        melhor = log.loc[log.loss_val.idxmin(), "epoca"]
        ax.axvline(melhor, color="#aaaaaa", ls=":", lw=1)
        ax.set(title=f"{tag} (melhor época: {melhor})", xlabel="época", ylabel="MSE (normalizado)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "curvas_treino.png")
    plt.close(fig)


def resumo_treino():
    print("\nÉpocas até parar / melhor época / tempo de treino (min):")
    for tag in ["global"] + [f"local_loja{i}" for i in range(1, 11)]:
        log = pd.read_csv(C.TAB_DIR / f"mtst_{tag}_log.csv")
        print(f"  {tag:12s} épocas={len(log):2d}  melhor={log.loc[log.loss_val.idxmin(), 'epoca']:2d}  "
              f"tempo={log.segundos.sum() / 60:.1f}")


if __name__ == "__main__":
    tabela_comparacao()
    mat = load_data()
    fig_previsoes(mat)
    fig_curvas()
    resumo_treino()
