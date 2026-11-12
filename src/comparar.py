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
COR_LR = "#6a8f3a"
NOME = {"real": "real", "naive_anual": "naive anual", "mtst_global": "MTST global",
        "mtst_local_todas": "MTST local (por loja)"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def tabela_comparacao():
    # v1: OneCycleLR de 30 épocas + early stopping (parou antes do decaimento do lr)
    # v2: OneCycleLR de 15 épocas fixas, sem early stopping
    partes = [pd.read_csv(C.TAB_DIR / "baselines.csv")]
    for nome in ["mtst_global", "mtst_local_todas"]:
        for v, suf in [("v1", "_v1"), ("v2", "")]:
            t = pd.read_csv(C.TAB_DIR / f"{nome}{suf}.csv")
            t["modelo"] = f"{nome}_{v}"
            partes.append(t)
    tab = pd.concat(partes, ignore_index=True)
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
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Janela 1 de teste (origem 2017-01-01)", y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(C.FIG_DIR / "previsoes_exemplo.png")
    plt.close(fig)


def fig_curvas():
    tags = ["global", "local_loja1", "local_loja2"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, tag in zip(axs, tags):
        log = pd.read_csv(C.TAB_DIR / f"mtst_{tag}_log.csv")
        ax.plot(log.epoca, log.loss_treino, color=COR["mtst_global"], marker="o", ms=3, label="treino")
        ax.plot(log.epoca, log.loss_val, color=COR["mtst_local_todas"], marker="o", ms=3, label="validação")
        melhor = log.loc[log.loss_val.idxmin(), "epoca"]
        ax.axvline(melhor, color="#aaaaaa", ls=":", lw=1)
        ax.set(title=f"{tag} (melhor época: {melhor})", xlabel="época", ylabel="MSE (normalizado)")
        ax2 = ax.twinx()  # lr ao final de cada época
        ax2.plot(log.epoca, log.lr, color=COR_LR, ls="--", lw=1.2, label="lr")
        ax2.set_ylabel("lr", color=COR_LR)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
    h1, l1 = axs[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axs[0].legend(h1 + h2, l1 + l2, frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "curvas_treino.png")
    plt.close(fig)


def resumo_treino():
    print("\nMelhor época (v1 / v2) e tempo de treino em min (v1 / v2):")
    tot = [0.0, 0.0]
    for tag in ["global"] + [f"local_loja{i}" for i in range(1, 11)]:
        l1 = pd.read_csv(C.TAB_DIR / f"mtst_{tag}_log_v1.csv")
        l2 = pd.read_csv(C.TAB_DIR / f"mtst_{tag}_log.csv")
        t1, t2 = l1.segundos.sum() / 60, l2.segundos.sum() / 60
        tot[0] += t1
        tot[1] += t2
        print(f"  {tag:12s} melhor: {l1.loc[l1.loss_val.idxmin(), 'epoca']:2d} / "
              f"{l2.loc[l2.loss_val.idxmin(), 'epoca']:2d}   tempo: {t1:.1f} / {t2:.1f}")
    print(f"  total treino (v1 / v2): {tot[0]:.1f} / {tot[1]:.1f} min")


if __name__ == "__main__":
    tabela_comparacao()
    mat = load_data()
    fig_previsoes(mat)
    fig_curvas()
    resumo_treino()
