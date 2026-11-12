"""Replicação do MTST no ETTh1 (experimento do artigo), seguindo run_longExp.py/exp_main.py oficiais.

  uv run python -m src.replicacao --horizonte 96
  uv run python -m src.replicacao --horizonte 96 --epocas 1 --amostra 0.1   # smoke test

Fiel ao oficial: Adam(lr=1e-4), MSE, early stopping (paciência 20) na loss de validação,
`lradj='type3'` e seed 2021. Detalhe herdado do oficial: com lradj != 'TST' o OneCycleLR é
construído mas nunca avançado; sua construção, porém, deixa lr=max_lr/25 (=4e-6) e beta1=0.95
no Adam até o fim da 1ª época, quando o type3 reajusta o lr para 1e-4 (beta1 continua 0.95).
Reproduzimos isso construindo o scheduler da mesma forma.
Métricas no teste: MSE/MAE na escala normalizada (média sobre todos os elementos).
"""
import argparse
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src import config as C
from src.ett import get_loader
from src.mtst_oficial.MTST import Model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixar_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_type3(lr0, epoca):  # utils/tools.py::adjust_learning_rate, 'type3' (epoca começa em 1)
    return lr0 if epoca < 3 else lr0 * (0.9 ** (epoca - 3))


def prever(model, x):
    return model(x.to(DEV))[0]  # (B, pred_len, 7); o MTST devolve (saida, draw_list, attn_list)


def vali(model, loader):
    """Como Exp_Main.vali: média das losses por batch (batches completos, drop_last)."""
    model.eval()
    perdas = []
    with torch.no_grad():
        for x, y in loader:
            perdas.append(((prever(model, x) - y.to(DEV)) ** 2).mean().item())
    model.train()
    return float(np.mean(perdas))


def testar(model, loader):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P.append(prever(model, x).cpu().numpy())
            Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    return float(np.mean((P - Y) ** 2)), float(np.mean(np.abs(P - Y))), len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, required=True, choices=C.HORIZONTES_ETTH1)
    ap.add_argument("--epocas", type=int, default=None, help="padrão: 100 (oficial)")
    ap.add_argument("--amostra", type=float, default=1.0, help="fração das janelas de treino (smoke test)")
    a = ap.parse_args()

    M, T = C.ETTH1["modelo"], C.ETTH1["treino"]
    H = a.horizonte
    epocas = a.epocas or T["train_epochs"]
    tag = f"etth1_h{H}" + ("" if a.epocas is None and a.amostra == 1.0 else "_smoke")
    torch.set_num_threads(os.cpu_count())  # como em train.py (Fase 2); o padrão do torch usa só os núcleos físicos
    fixar_seed(T["seed"])
    t_ini = time.time()
    dev_nome = torch.cuda.get_device_name(0) if DEV.type == "cuda" else f"cpu ({torch.get_num_threads()} threads)"
    print(f"[{tag}] device: {dev_nome}")

    _, tr = get_loader("train", M["seq_len"], H, T["batch_size"], a.amostra)
    _, va = get_loader("val", M["seq_len"], H, T["batch_size"])
    _, te = get_loader("test", M["seq_len"], H, T["batch_size"])
    print(f"[{tag}] batches treino={len(tr)} val={len(va)} teste={len(te)}")

    model = Model(SimpleNamespace(pred_len=H, **M)).float().to(DEV)
    print(f"[{tag}] parâmetros: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = torch.optim.Adam(model.parameters(), lr=T["learning_rate"], weight_decay=T["l2"])
    # Construído e NUNCA avançado, como no oficial (ver docstring).
    torch.optim.lr_scheduler.OneCycleLR(opt, steps_per_epoch=len(tr), pct_start=T["pct_start"],
                                        epochs=epocas, max_lr=T["learning_rate"])

    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = C.CKPT_DIR / f"{tag}.pt"
    melhor, melhor_ep, sem_melhora, log = float("inf"), 0, 0, []

    for ep in range(1, epocas + 1):
        t0 = time.time()
        lr_ep = opt.param_groups[0]["lr"]
        model.train()
        perdas = []
        for i, (x, y) in enumerate(tr):
            opt.zero_grad()
            y = y.to(DEV)
            loss = ((prever(model, x) - y) ** 2).mean()
            if ep == 1 and i == 0:
                print(f"[{tag}] shapes: x={tuple(x.shape)} y={tuple(y.shape)} saída={tuple(prever(model, x).shape)}")
            loss.backward()
            opt.step()
            perdas.append(loss.item())
        l_tr, l_va = float(np.mean(perdas)), vali(model, va)
        dt = time.time() - t0
        marca = ""
        if l_va < melhor:
            melhor, melhor_ep, sem_melhora, marca = l_va, ep, 0, " *"
            torch.save(model.state_dict(), ckpt)
        else:
            sem_melhora += 1
        print(f"[{tag}] época {ep:03d} | treino {l_tr:.4f} | val {l_va:.4f} | lr {lr_ep:.2e} | {dt:.0f}s{marca}")
        log.append(dict(epoca=ep, loss_treino=l_tr, loss_val=l_va, lr=lr_ep, segundos=dt))
        if sem_melhora >= T["patience"]:
            print(f"[{tag}] early stopping")
            break
        for g in opt.param_groups:  # adjust_learning_rate(epoch + 1)
            g["lr"] = lr_type3(T["learning_rate"], ep)

    pd.DataFrame(log).to_csv(C.TAB_DIR / f"{tag}_log.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=DEV))
    mse, mae, n = testar(model, te)
    print(f"[{tag}] TESTE (n={n} janelas): mse={mse:.4f} mae={mae:.4f}")
    linha = dict(horizonte=H, mse=mse, mae=mae, melhor_epoca=melhor_ep, epocas_rodadas=len(log),
                 melhor_val=melhor, tempo_min=(time.time() - t_ini) / 60, device=dev_nome)
    if tag.endswith("_smoke"):  # smoke não entra na tabela de resultados
        pd.DataFrame([linha]).to_csv(C.TAB_DIR / f"{tag}.csv", index=False)
    else:
        registrar(linha)


def registrar(linha):
    """Atualiza results/tables/etth1_resultados.csv (uma linha por horizonte) com a comparação ao artigo."""
    art = pd.read_csv(C.TAB_DIR / "etth1_artigo.csv").set_index("horizonte")
    arq = C.TAB_DIR / "etth1_resultados.csv"
    tab = pd.read_csv(arq) if arq.exists() else pd.DataFrame()
    tab = pd.concat([tab[tab.horizonte != linha["horizonte"]] if len(tab) else tab,
                     pd.DataFrame([linha])], ignore_index=True).sort_values("horizonte")
    for m in ("mse", "mae"):
        ref = tab.horizonte.map(art[m])
        tab[f"{m}_artigo"] = ref
        tab[f"{m}_dif_abs"] = tab[m] - ref
        tab[f"{m}_dif_rel_pct"] = 100 * (tab[m] - ref) / ref
    tab.to_csv(arq, index=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
