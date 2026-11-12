"""Loader do ETTh1 que reproduz o `Dataset_ETT_hour` do código oficial do MTST.

Origem: https://github.com/networkslab/MTST, `data_provider/data_loader.py` (commit d52b6b5),
classe `Dataset_ETT_hour` + `data_provider/data_factory.py` (flags de DataLoader).
Dados: https://github.com/zhouhaoyi/ETDataset (ETT-small/ETTh1.csv), 17.420 linhas.

Reproduzido exatamente:
  - fronteiras (12/4/4 meses de 30 dias, em horas):
      treino [0, 8640), val [8640-seq_len, 11520), teste [11520-seq_len, 14400)
    (val e teste começam seq_len antes, para a 1ª janela já ter histórico completo;
     as últimas 3020 linhas do CSV não são usadas, como no oficial)
  - StandardScaler ajustado só em df[0:8640], aplicado a toda a série
  - features 'M': as 7 colunas (HUFL..OT); janela x = seq_len, y = pred_len (label_len=0)
  - len = len(data) - seq_len - pred_len + 1
  - DataLoader: treino shuffle+drop_last; val e teste sem shuffle, drop_last=True (data_factory)

Diferenças (sem efeito nos resultados do MTST):
  - StandardScaler do sklearn substituído por numpy equivalente (média e desvio populacional,
    ddof=0; desvio 0 -> 1), para não adicionar dependência.
  - Marcas de tempo (x_mark/y_mark) não são geradas: o MTST não as usa.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import config as C

_MES = 30 * 24  # 720 h


class ETTh1(Dataset):
    def __init__(self, flag="train", seq_len=336, pred_len=96, path=None, features="M"):
        assert flag in ("train", "val", "test")
        t = {"train": 0, "val": 1, "test": 2}[flag]
        self.seq_len, self.pred_len = seq_len, pred_len

        df = pd.read_csv(path or C.ETTH1_CSV)
        b1s = [0, 12 * _MES - seq_len, 12 * _MES + 4 * _MES - seq_len]
        b2s = [12 * _MES, 12 * _MES + 4 * _MES, 12 * _MES + 8 * _MES]

        vals = df[df.columns[1:]].values if features in ("M", "MS") else df[["OT"]].values
        treino = vals[b1s[0]:b2s[0]]
        self.mean = treino.mean(axis=0)
        self.scale = treino.std(axis=0)  # ddof=0, igual ao sklearn
        self.scale[self.scale == 0.0] = 1.0
        data = (vals - self.mean) / self.scale
        self.data = data[b1s[t]:b2s[t]]

    def __getitem__(self, i):
        x = self.data[i:i + self.seq_len]
        y = self.data[i + self.seq_len:i + self.seq_len + self.pred_len]  # label_len = 0
        return x.astype(np.float32), y.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1


def get_loader(flag, seq_len, pred_len, batch_size, amostra=1.0, seed=0):
    ds = ETTh1(flag, seq_len, pred_len)
    if amostra < 1.0 and flag == "train":  # só para smoke tests: usa fração das janelas
        n = max(batch_size, int(len(ds) * amostra))
        ds = torch.utils.data.Subset(ds, range(n))
    treino = flag == "train"
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=treino, drop_last=True, num_workers=0)
