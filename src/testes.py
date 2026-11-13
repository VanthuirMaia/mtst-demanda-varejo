"""Testes de Wilcoxon pareados por série (70 séries): MAE por série médio nas 4 janelas de teste,
a partir das previsões da seed 42 (pred_m5_*.npy). Pares: PatchTST x MTST, PatchTST x ARIMA,
MTST x ARIMA. Saída: results/tables/m5_testes.csv.
Só a seed 42 tem previsões salvas (src/final.py guarda apenas essa)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import config as C
from src.data import get_test_origins, load_data

MODELOS = {"patchtst": "patchtst_global", "mtst": "mtst_global", "arima": "arima"}
PARES = [("patchtst", "mtst"), ("patchtst", "arima"), ("mtst", "arima")]


def mae_por_serie(mat):
    """-> dict nome -> array (70,): MAE por série, média das 4 janelas."""
    vals = mat.to_numpy(dtype="float64")
    reais = np.stack([vals[mat.index.get_loc(o):mat.index.get_loc(o) + C.HORIZON].T
                      for o in get_test_origins()])  # (4, S, H)
    out = {}
    for k, nome in MODELOS.items():
        p = np.load(C.PROCESSED_DIR / f"pred_{C.PREFIX}{nome}.npy")  # (4, S, H)
        assert p.shape == reais.shape, (nome, p.shape, reais.shape)
        out[k] = np.abs(reais - p).mean(axis=2).mean(axis=0)
    return out


def main():
    mat = load_data()
    mae = mae_por_serie(mat)
    linhas = []
    for a, b in PARES:
        d = mae[a] - mae[b]
        r = wilcoxon(mae[a], mae[b])
        linhas.append({
            "modelo_a": MODELOS[a], "modelo_b": MODELOS[b], "seeds": "42", "n_series": len(d),
            "mae_medio_a": mae[a].mean(), "mae_medio_b": mae[b].mean(),
            "mediana_dif_a_menos_b": np.median(d),
            "estatistica_W": r.statistic, "p_valor": r.pvalue,
            "vitorias_a": int((d < 0).sum()), "vitorias_b": int((d > 0).sum()), "empates": int((d == 0).sum()),
        })
    tab = pd.DataFrame(linhas)
    # Holm (3 comparações)
    ordem = tab.p_valor.argsort().to_numpy()
    holm = np.empty(len(tab))
    maxi = 0.0
    for i, j in enumerate(ordem):
        maxi = max(maxi, min(1.0, (len(tab) - i) * tab.p_valor[j]))
        holm[j] = maxi
    tab["p_holm"] = holm
    tab.to_csv(C.TAB_DIR / f"{C.PREFIX}testes.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return tab


if __name__ == "__main__":
    main()
