"""Treino e avaliação do MTST (global ou local).

Exemplos:
  uv run python -m src.train --modo global
  uv run python -m src.train --modo local --loja 3
  uv run python -m src.train --modo local_todas
  uv run python -m src.train --modo global --epocas 1 --amostra 0.1   # teste rápido
"""
import argparse
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config as C
from src.data import get_test_origins, load_data, make_test_window, make_windows, selecionar_series
from src.metrics import evaluate
from src.mtst_oficial.MTST import Model


def fixar_seed(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def criar_modelo():
    return Model(SimpleNamespace(**C.MTST))


def contar_parametros(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prever(model, x):
    """x: (B, lookback) -> (B, horizon). O modelo oficial usa [B, L, canais] com 1 canal."""
    saida = model(x.unsqueeze(-1))[0]  # (B, horizon, 1)
    return saida.squeeze(-1)


def prever_em_lotes(model, X, batch=512):
    model.eval()
    X = torch.from_numpy(X)
    with torch.no_grad():
        return torch.cat([prever(model, X[i:i + batch]) for i in range(0, len(X), batch)]).numpy()


def loss_val(model, X, y, batch=512):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            pred = prever(model, X[i:i + batch])
            total += ((pred - y[i:i + batch]) ** 2).mean(dim=1).sum().item()
    return total / len(X)


def treinar(mat, tag, epocas, verbose=True):
    """Treina em `mat` (DataFrame T x séries). Devolve o modelo com os melhores pesos na validação."""
    T = C.TREINO
    Xtr, ytr = make_windows(mat, mat.index[0], C.TRAIN_END, stride=T["stride_treino"])
    Xva, yva = make_windows(mat, pd.Timestamp(C.TRAIN_END) + pd.Timedelta(days=1), C.VAL_END,
                            stride=T["stride_val"])
    Xtr, ytr, Xva, yva = (torch.from_numpy(a) for a in (Xtr, ytr, Xva, yva))
    print(f"[{tag}] amostras treino={len(Xtr)} val={len(Xva)} séries={mat.shape[1]}")

    model = criar_modelo()
    print(f"[{tag}] parâmetros: {contar_parametros(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=T["lr"])
    gen = torch.Generator().manual_seed(C.SEED)

    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = C.CKPT_DIR / f"mtst_{tag}.pt"
    melhor, sem_melhora, log = float("inf"), 0, []

    for ep in range(1, epocas + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        soma = 0.0
        barra = tqdm(range(0, len(perm), T["batch"]), desc=f"época {ep}", leave=False, disable=not verbose, mininterval=60)
        for i in barra:
            idx = perm[i:i + T["batch"]]
            opt.zero_grad()
            loss = ((prever(model, Xtr[idx]) - ytr[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            soma += loss.item() * len(idx)
        l_tr = soma / len(Xtr)
        l_va = loss_val(model, Xva, yva)
        dt = time.time() - t0
        marca = ""
        if l_va < melhor:
            melhor, sem_melhora, marca = l_va, 0, " *"
            torch.save(model.state_dict(), ckpt)
        else:
            sem_melhora += 1
        print(f"[{tag}] época {ep:02d} | treino {l_tr:.3f} | val {l_va:.3f} | {dt:.0f}s{marca}")
        log.append({"epoca": ep, "loss_treino": l_tr, "loss_val": l_va, "segundos": dt})
        if sem_melhora >= T["paciencia"]:
            print(f"[{tag}] early stopping (paciência {T['paciencia']})")
            break

    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(C.TAB_DIR / f"mtst_{tag}_log.csv", index=False)
    model.load_state_dict(torch.load(ckpt))
    return model


def prever_teste(model, mat):
    """Para cada origem de teste: (y_true, y_pred clipado em >= 0), ambos (n_series, horizon)."""
    saida = []
    for origem in get_test_origins():
        X, y = make_test_window(mat, origem)
        pred = np.clip(prever_em_lotes(model, X), 0, None)
        saida.append((y, pred))
    return saida


def tabela(nome, resultados, y_train):
    """resultados: lista de (y_true, y_pred) por janela. Mesmo formato do baselines.csv."""
    linhas = []
    for k, (origem, (y, p)) in enumerate(zip(get_test_origins(), resultados), start=1):
        linhas.append({"modelo": nome, "janela": k, "origem": origem.date(), **evaluate(y, p, y_train)})
    tab = pd.DataFrame(linhas)
    med = tab[["mae", "rmse", "smape", "mase"]].mean().to_frame().T
    med.insert(0, "modelo", nome)
    med.insert(1, "janela", "media")
    return pd.concat([tab, med], ignore_index=True)


def y_treino(mat):
    n = mat.index.searchsorted(pd.Timestamp(C.TRAIN_END), side="right")
    return mat.to_numpy(dtype="float64")[:n].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["global", "local", "local_todas"], required=True)
    ap.add_argument("--loja", type=int, help="loja (1-10) para --modo local")
    ap.add_argument("--epocas", type=int, default=C.TREINO["epocas"])
    ap.add_argument("--amostra", type=float, default=1.0, help="fração de séries (teste rápido)")
    args = ap.parse_args()
    if args.modo == "local" and args.loja is None:
        ap.error("--modo local exige --loja N")

    torch.set_num_threads(os.cpu_count())
    fixar_seed()
    mat = load_data()

    if args.modo == "global":
        sub = selecionar_series(mat, frac=args.amostra)
        model = treinar(sub, "global", args.epocas)
        res, yt, nome = prever_teste(model, sub), y_treino(sub), "mtst_global"
    elif args.modo == "local":
        sub = selecionar_series(mat, loja=args.loja, frac=args.amostra)
        tag = f"local_loja{args.loja}"
        model = treinar(sub, tag, args.epocas)
        res, yt, nome = prever_teste(model, sub), y_treino(sub), f"mtst_{tag}"
    else:  # um modelo por loja; métricas sobre todas as séries juntas
        por_loja = []
        for loja in range(1, 11):
            fixar_seed()
            sub = selecionar_series(mat, loja=loja, frac=args.amostra)
            model = treinar(sub, f"local_loja{loja}", args.epocas)
            por_loja.append((prever_teste(model, sub), y_treino(sub)))
        res = [(np.concatenate([r[0][k][0] for r in por_loja]),
                np.concatenate([r[0][k][1] for r in por_loja])) for k in range(C.N_TEST_WINDOWS)]
        yt = np.concatenate([r[1] for r in por_loja])
        nome = "mtst_local_todas"

    tab = tabela(nome, res, yt)
    saida = C.TAB_DIR / f"{nome}.csv"
    tab.to_csv(saida, index=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("salvo em", saida)


if __name__ == "__main__":
    main()
