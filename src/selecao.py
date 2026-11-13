"""Seleção de parâmetros em grade (MTST e PatchTST globais), usando SÓ treino e validação
(nunca o teste). Protocolo único: 15 épocas fixas, OneCycleLR max_lr 5e-4, seed 42, melhor
checkpoint pelo MSE de validação na escala normalizada.

Retomável: configs já presentes em m5_selecao.csv são puladas.
  uv run python -m src.selecao
  uv run python -m src.selecao --smoke    # 1 config por modelo, 1 época, saída em results/smoke/
"""
import argparse
import itertools

import pandas as pd

from src import config as C
from src.backup import salvar_backup

COLUNAS = ["modelo", "config", "val_mse_norm", "melhor_epoca", "n_parametros", "segundos"]


def id_config(patch, d_model, dropout):
    p = "-".join(str(x) for x in patch) if isinstance(patch, (tuple, list)) else str(patch)
    return f"p{p}_d{d_model}_dr{dropout}"


def configs(modelo):
    g = C.GRADE[modelo]
    for patch, d, dr in itertools.product(g["patch"], g["d_model"], g["dropout"]):
        yield id_config(patch, d, dr), patch, d, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 config/modelo, 1 época, results/smoke/")
    ap.add_argument("--backup", metavar="DIR", help="copia m5_* e pred_m5_* para DIR a cada config")
    ap.add_argument("--modelos", nargs="+", default=list(C.GRADE), choices=list(C.GRADE))
    args = ap.parse_args()
    if args.smoke:
        C.usar_smoke()
    epocas = 1 if args.smoke else C.TREINO["epocas"]

    from src.data import load_data
    from src.train import DEV, treinar
    mat = load_data()
    C.imprimir_splits()
    print(f"device: {DEV} | épocas: {epocas} | stride de treino: {C.TREINO['stride_treino']}")

    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    saida = C.TAB_DIR / f"{C.PREFIX}selecao.csv"
    feito = pd.read_csv(saida) if saida.exists() else pd.DataFrame(columns=COLUNAS)
    ja = set(zip(feito.modelo, feito.config))

    for modelo in args.modelos:
        lista = list(configs(modelo))
        if args.smoke:
            lista = lista[:1]
        for i, (cid, patch, d, dr) in enumerate(lista, start=1):
            if (modelo, cid) in ja:
                print(f"[{modelo}] {cid}: já no CSV, pulando ({i}/{len(lista)})")
                continue
            print(f"[{modelo}] {cid} ({i}/{len(lista)})")
            r = treinar(mat, f"{modelo}:{cid}", epocas, modelo, patch, d, dr, seed=C.SEED)
            linha = dict(modelo=modelo, config=cid, val_mse_norm=r["melhor_val"],
                         melhor_epoca=r["melhor_ep"], n_parametros=r["n_params"],
                         segundos=round(r["segundos"], 1))
            feito = pd.concat([feito, pd.DataFrame([linha])], ignore_index=True)
            feito.to_csv(saida, index=False)  # grava a cada config (retomada após desconexão)
            ja.add((modelo, cid))
            salvar_backup(args.backup)

    print("\n" + feito.sort_values(["modelo", "val_mse_norm"]).to_string(index=False))
    print("salvo em", saida)


if __name__ == "__main__":
    main()
