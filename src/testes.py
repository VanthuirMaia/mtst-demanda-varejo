"""Testes de Wilcoxon pareados por série (70 séries) sobre o MAE por série médio nas 4 janelas
de teste. Pares: PatchTST x MTST, PatchTST x ARIMA, MTST x ARIMA. Duas versões:
  - seed 42 (pred_m5_{modelo}_s42.npy ou pred_m5_{modelo}.npy);
  - média das seeds 42, 43 e 44: para cada série, média dos MAE das seeds (não é o MAE da média
    das previsões); usada se os pred_m5_{modelo}_s{seed}.npy de todas as seeds existirem.
Saída: results/tables/m5_testes.csv."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import config as C
from src.data import get_test_origins, load_data

MODELOS = {"patchtst": "patchtst_global", "mtst": "mtst_global", "arima": "arima"}
PARES = [("patchtst", "mtst"), ("patchtst", "arima"), ("mtst", "arima")]


def _reais(mat):
    vals = mat.to_numpy(dtype="float64")
    return np.stack([vals[mat.index.get_loc(o):mat.index.get_loc(o) + C.HORIZON].T
                     for o in get_test_origins()])  # (4, S, H)


def _mae(reais, arq):
    p = np.load(arq)
    assert p.shape == reais.shape, (arq, p.shape, reais.shape)
    return np.abs(reais - p).mean(axis=2).mean(axis=0)  # (S,)


def _arq(nome, seed):
    d = C.PROCESSED_DIR
    if nome == "arima":
        return d / f"pred_{C.PREFIX}arima.npy"
    novo = d / f"pred_{C.PREFIX}{nome}_s{seed}.npy"
    if novo.exists():
        return novo
    return d / f"pred_{C.PREFIX}{nome}.npy" if seed == C.SEED else novo  # legado: só seed 42


def mae_por_serie(mat, seeds):
    """-> dict nome -> array (70,); None se faltar algum arquivo. Média dos MAE entre `seeds`."""
    reais = _reais(mat)
    out = {}
    for k, nome in MODELOS.items():
        arqs = [_arq(nome, s) for s in (seeds if k != "arima" else (C.SEED,))]
        if not all(a.exists() for a in arqs):
            return None
        out[k] = np.mean([_mae(reais, a) for a in arqs], axis=0)
    return out


def holm(p):
    p = np.asarray(p, dtype=float)
    ordem = np.argsort(p)
    out, maxi = np.empty(len(p)), 0.0
    for i, j in enumerate(ordem):
        maxi = max(maxi, min(1.0, (len(p) - i) * p[j]))
        out[j] = maxi
    return out


def testar(mae, rotulo):
    linhas = []
    for a, b in PARES:
        d = mae[a] - mae[b]
        r = wilcoxon(mae[a], mae[b])
        linhas.append({
            "modelo_a": MODELOS[a], "modelo_b": MODELOS[b], "seeds": rotulo, "n_series": len(d),
            "mae_medio_a": mae[a].mean(), "mae_medio_b": mae[b].mean(),
            "mediana_dif_a_menos_b": np.median(d), "estatistica_W": r.statistic, "p_valor": r.pvalue,
            "vitorias_a": int((d < 0).sum()), "vitorias_b": int((d > 0).sum()), "empates": int((d == 0).sum()),
        })
    t = pd.DataFrame(linhas)
    t["p_holm"] = holm(t.p_valor)  # correção para as 3 comparações de cada versão
    return t


def main():
    mat = load_data()
    partes = []
    m42 = mae_por_serie(mat, (42,))
    assert m42 is not None, "faltam previsões da seed 42"
    partes.append(testar(m42, "42"))
    m3 = mae_por_serie(mat, C.SEEDS_FINAL)
    if m3 is None:
        print("previsões das seeds 43 e 44 não encontradas: só o teste da seed 42")
    else:
        partes.append(testar(m3, "media_42-43-44"))
    tab = pd.concat(partes, ignore_index=True)
    tab.to_csv(C.TAB_DIR / f"{C.PREFIX}testes.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return tab


if __name__ == "__main__":
    main()
