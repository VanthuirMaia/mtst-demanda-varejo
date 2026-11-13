"""Treino final: melhor config de cada modelo (por m5_selecao.csv), seeds 42, 43 e 44, avaliação
nas 4 janelas de teste (previsões desnormalizadas e clipadas em >= 0). Retomável.
  uv run python -m src.final
  uv run python -m src.final --smoke    # 1 época, 1 seed, results/smoke/ (usa o selecao do smoke)
"""
import argparse
import shutil

import numpy as np
import pandas as pd

from src import config as C
from src.metrics import METRICAS

MODELOS = ["patchtst_global", "mtst_global"]


def melhor_config(sel, modelo):
    """Linha de menor val_mse_norm do modelo -> (config_id, patch, d_model, dropout)."""
    from src.selecao import configs
    s = sel[sel.modelo == modelo]
    assert len(s), f"{modelo} sem linhas em m5_selecao.csv"
    cid = s.loc[s.val_mse_norm.idxmin(), "config"]
    for c, patch, d, dr in configs(modelo):
        if c == cid:
            return c, patch, d, dr
    raise ValueError(f"config {cid} fora da grade")


def tabela_comparacao(final, base):
    """naive/ARIMA (média das 4 janelas) + Transformers (média e desvio entre seeds,
    cada seed já média das 4 janelas; desvio amostral, ddof=1). Colunas <m> e <m>_dp."""
    linhas = []
    for nome, g in base[base.janela == "media"].groupby("modelo", sort=False):
        linhas.append({"modelo": nome, "n_seeds": np.nan,
                       **{m: g[m].iloc[0] for m in METRICAS}, **{f"{m}_dp": np.nan for m in METRICAS}})
    for modelo in MODELOS:
        f = final[(final.modelo == modelo) & (final.janela == "media")]
        if f.empty:
            continue
        linhas.append({"modelo": modelo, "config": f.config.iloc[0], "n_seeds": len(f),
                       **{m: f[m].mean() for m in METRICAS},
                       **{f"{m}_dp": f[m].std(ddof=1) if len(f) > 1 else np.nan for m in METRICAS}})
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    base_tab = C.TAB_DIR / f"{C.PREFIX}baselines.csv"
    base_arima = C.PROCESSED_DIR / f"pred_{C.PREFIX}arima.npy"
    if args.smoke:
        C.usar_smoke()
        C.TAB_DIR.mkdir(parents=True, exist_ok=True)
        C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(base_tab, C.TAB_DIR)
        if base_arima.exists():
            shutil.copy(base_arima, C.PROCESSED_DIR)
    seeds = (C.SEEDS_FINAL[0],) if args.smoke else C.SEEDS_FINAL
    epocas = 1 if args.smoke else C.TREINO["epocas"]

    from src.comparar import fig_curvas, fig_previsoes
    from src.data import load_data
    from src.train import DEV, prever_teste, tabela, treinar, y_treino
    mat = load_data()
    C.imprimir_splits()
    print(f"device: {DEV} | épocas: {epocas} | seeds: {seeds}")
    for d in (C.TAB_DIR, C.FIG_DIR, C.PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(C.TAB_DIR / f"{C.PREFIX}selecao.csv")
    f_final = C.TAB_DIR / f"{C.PREFIX}final.csv"
    final = pd.read_csv(f_final) if f_final.exists() else pd.DataFrame()
    yt = y_treino(mat)

    for modelo in MODELOS:
        cid, patch, d, dr = melhor_config(sel, modelo)
        print(f"\n=== {modelo}: melhor config {cid} ===")
        for seed in seeds:
            arq_pred = C.PROCESSED_DIR / f"pred_{C.PREFIX}{modelo}.npy"  # seed 42: figuras
            arq_log = C.TAB_DIR / f"{C.PREFIX}{modelo}_log.csv"
            feito = (not final.empty and ((final.modelo == modelo) & (final.seed == seed)).any())
            if feito and (seed != C.SEED or (arq_pred.exists() and arq_log.exists())):
                print(f"[{modelo}] seed {seed}: já no CSV, pulando")
                continue
            r = treinar(mat, f"{modelo}:s{seed}", epocas, modelo, patch, d, dr, seed=seed)
            res = prever_teste(r["model"], mat, r["mu"], r["sd"])
            tab = tabela(modelo, res, yt)
            tab.insert(1, "config", cid)
            tab.insert(2, "seed", seed)
            tab["melhor_epoca"] = r["melhor_ep"]
            tab["segundos_treino"] = round(r["segundos"], 1)
            if seed == C.SEED:
                np.save(arq_pred, np.stack([p for _, p in res]))
                pd.DataFrame(r["log"]).to_csv(arq_log, index=False)
            if not final.empty:
                final = final[~((final.modelo == modelo) & (final.seed == seed))]
            final = pd.concat([final, tab], ignore_index=True)
            final.to_csv(f_final, index=False)  # grava a cada seed (retomada)

    base = pd.read_csv(C.TAB_DIR / f"{C.PREFIX}baselines.csv")
    comp = tabela_comparacao(final, base)
    comp.to_csv(C.TAB_DIR / f"{C.PREFIX}comparacao.csv", index=False)
    print("\n" + comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fig_previsoes(mat)
    fig_curvas()
    print("figuras em", C.FIG_DIR)


if __name__ == "__main__":
    main()
