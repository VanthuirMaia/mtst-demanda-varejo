"""Análise exploratória. Gera figuras em results/figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.data import get_test_origins, load_data

COR = "#2a6fb0"      # série principal
COR_REF = "#8a8a8a"  # referências / cortes
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def salvar(fig, nome):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / nome)
    plt.close(fig)
    print("figura:", C.FIG_DIR / nome)


def fig_total(mat):
    total = mat.sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(total.index, total, color=COR, lw=0.6, alpha=0.5, label="diário")
    ax.plot(total.index, total.rolling(30).mean(), color=COR, lw=2, label="média móvel 30d")
    t_end, v_end = C.to_ts(C.TRAIN_END), C.to_ts(C.VAL_END)
    origens = get_test_origins()
    t_ini = origens[0]
    t_fim = origens[-1] + pd.Timedelta(days=C.HORIZON - 1)
    ax.axvspan(total.index[0], t_end, color="#2a6fb0", alpha=0.05)
    ax.axvspan(t_end, v_end, color="#e0a030", alpha=0.12)
    ax.axvspan(t_ini, t_fim, color="#c0392b", alpha=0.10)
    for d in [t_end, v_end] + origens[1:]:
        ax.axvline(d, color=COR_REF, lw=1, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(total.index[0], ymax, " treino", va="top", color="#444")
    ax.text(t_end, ymax, " val", va="top", color="#444")
    ax.text(t_ini, ymax, " teste (4x90d)", va="top", color="#444")
    ax.set(title="Vendas totais diárias (500 séries)", ylabel="unidades/dia")
    ax.legend(loc="lower right", frameon=False)
    salvar(fig, "vendas_totais.png")


def fig_semana(mat):
    rel = mat / mat.mean()  # normaliza cada série pela sua média
    perfil = rel.groupby(mat.index.day_of_week).mean().mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(DIAS, perfil.values, color=COR, width=0.6)
    ax.axhline(1, color=COR_REF, lw=1)
    ax.set(title="Perfil por dia da semana", ylabel="venda / média da série")
    ax.set_ylim(0.6, perfil.max() * 1.08)
    salvar(fig, "perfil_semanal.png")


def fig_mes(mat):
    # divide cada dia pela média da série no mesmo ano -> remove a tendência
    rel = mat / mat.groupby(mat.index.year).transform("mean")
    perfil = rel.groupby(mat.index.month).mean().mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(MESES, perfil.values, color=COR, width=0.65)
    ax.axhline(1, color=COR_REF, lw=1)
    ax.set(title="Perfil mensal (sazonalidade anual)", ylabel="venda / média anual da série")
    ax.set_ylim(0.6, perfil.max() * 1.08)
    salvar(fig, "perfil_mensal.png")


def fig_heatmap(mat):
    media = mat.mean()
    grade = np.zeros((10, 50))
    for col, v in media.items():
        s, i = (int(p[1:]) for p in col.split("_"))
        grade[s - 1, i - 1] = v
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(grade, aspect="auto", cmap="Blues")
    ax.grid(False)
    ax.set(title="Venda média diária por loja x item", xlabel="item", ylabel="loja")
    ax.set_xticks(range(0, 50, 4), labels=range(1, 51, 4))
    ax.set_yticks(range(10), labels=range(1, 11))
    fig.colorbar(im, ax=ax, label="unidades/dia")
    salvar(fig, "heatmap_loja_item.png")


def estatisticas(mat):
    media, dp = mat.mean(), mat.std()
    cv = dp / media
    print("\n--- Estatísticas por série ---")
    print(f"Escala (média diária): média={media.mean():.1f}  min={media.min():.1f}  "
          f"max={media.max():.1f}  razão max/min={media.max() / media.min():.1f}")
    print(f"Coef. de variação:     média={cv.mean():.2f}  min={cv.min():.2f}  max={cv.max():.2f}")
    print(f"Zeros: {(mat == 0).sum().sum()} observações ({(mat == 0).mean().mean():.4%})")
    print(f"Total diário: média={mat.sum(axis=1).mean():.0f}")


def main():
    mat = load_data()
    fig_total(mat)
    fig_semana(mat)
    fig_mes(mat)
    fig_heatmap(mat)
    estatisticas(mat)


if __name__ == "__main__":
    main()
