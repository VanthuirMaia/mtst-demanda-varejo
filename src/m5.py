"""Leitura do M5 (Walmart) agregado por loja x departamento."""
import pandas as pd

from src import config as C

CACHE = C.PROCESSED_DIR / "m5_store_dept.csv"


def _datas_calendario():
    cal = pd.read_csv(C.M5_CALENDAR_CSV, usecols=["date", "d"], parse_dates=["date"])
    return cal.set_index("d")["date"]


def load_m5(verbose=True):
    """Devolve DataFrame (1941 dias x 70 séries) com vendas somadas por store_id x dept_id.
    Colunas '{store}_{dept}' (ex.: 'CA_1_HOBBIES_1'), ordenadas por loja e departamento."""
    for f in (C.M5_SALES_CSV, C.M5_CALENDAR_CSV):
        if not f.exists():
            raise FileNotFoundError(
                f"{f} não encontrado. Baixe sales_train_evaluation.csv e calendar.csv do M5 "
                "(Kaggle: m5-forecasting-accuracy) e coloque em data/raw/m5/."
            )
    if CACHE.exists() and CACHE.stat().st_mtime >= C.M5_SALES_CSV.stat().st_mtime:
        mat = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    else:
        cab = pd.read_csv(C.M5_SALES_CSV, nrows=0).columns
        dcols = [c for c in cab if c.startswith("d_")]
        df = pd.read_csv(C.M5_SALES_CSV, usecols=["store_id", "dept_id"] + dcols,
                         dtype={c: "int32" for c in dcols})
        soma = df.groupby(["store_id", "dept_id"])[dcols].sum()  # (70, 1941)
        soma.index = [f"{s}_{d}" for s, d in soma.index]
        mat = soma.T
        mat.index = pd.DatetimeIndex(_datas_calendario().loc[mat.index].values, name="date")
        C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        mat.to_csv(CACHE)
    mat = mat.astype("float32")

    # Validações
    assert mat.shape == (C.M5_N_DAYS, 70), f"shape inesperado: {mat.shape}"
    assert not mat.isna().any().any(), "Há NaN na matriz"
    esperado = pd.date_range(mat.index.min(), mat.index.max(), freq="D")
    assert mat.index.equals(esperado), "datas não são diárias e contíguas"
    assert mat.index[0] == pd.Timestamp(C.M5_FIRST_DAY), "primeiro dia diferente do config"
    if verbose:
        print(f"Dados M5: shape={mat.shape} (dias x séries), "
              f"período {mat.index.min().date()} a {mat.index.max().date()}")
    return mat


def dias_quase_zero(mat, frac=0.05):
    """Dias em que a venda total (70 séries) é < frac x mediana do total diário.
    -> Series (data -> total)."""
    total = mat.sum(axis=1)
    return total[total < frac * total.median()]


def lojas_fechadas(mat):
    """Dias com venda zero em uma loja inteira (soma dos 7 departamentos), exceto os dias em
    que o total geral também é ~0. -> DataFrame (loja, n_dias, primeiro, ultimo)."""
    lojas = mat.T.groupby(lambda c: "_".join(c.split("_")[:2])).sum().T  # store_id = estado_n
    ignorar = dias_quase_zero(mat).index
    z = (lojas == 0).drop(index=ignorar, errors="ignore")
    linhas = []
    for loja in lojas.columns:
        d = z.index[z[loja]]
        if len(d):
            linhas.append({"loja": loja, "n_dias": len(d), "primeiro": d.min().date(),
                           "ultimo": d.max().date()})
    return pd.DataFrame(linhas, columns=["loja", "n_dias", "primeiro", "ultimo"])


if __name__ == "__main__":
    m = load_m5()
    print("\nDias com venda total ~0 (< 5% da mediana):")
    print(dias_quase_zero(m).to_string())
    print("\nLojas com dia inteiro de venda zero (fora os dias acima):")
    print(lojas_fechadas(m).to_string(index=False))
