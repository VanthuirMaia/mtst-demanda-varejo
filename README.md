# MTST em previsão de demanda de varejo: modelo global vs. local

Projeto da disciplina Previsão de Séries Temporais (mestrado).

## Ideia central

**Problema.** Prever 90 dias de vendas diárias de 500 séries (10 lojas x 50 itens) do
Kaggle *Store Item Demand Forecasting Challenge*. As séries têm sazonalidade semanal
forte, sazonalidade anual e tendência.

**Por que MTST.** O *Multi-resolution Time-Series Transformer* (Zhang et al., AISTATS 2024)
divide a série em patches de **vários tamanhos**, cada tamanho em um ramo paralelo do
Transformer. Patches pequenos captam a sazonalidade semanal; médios, a mensal; grandes,
a anual, tudo ao mesmo tempo. Usa também *positional encoding relativo*, que lida melhor
com padrões periódicos de períodos diferentes do que o encoding absoluto.

**Pergunta de pesquisa.** Em demanda de varejo, vale mais treinar **um modelo global**
(uma rede para as 500 séries, channel-independent) ou **modelos locais** (um por série,
ou por loja/item)? O global compartilha padrões e tem mais dados; o local se adapta a
cada série, mas tem poucas amostras.

## Dataset

[Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)
(Kaggle). Colunas: `date, store, item, sales`. 10 lojas x 50 itens, diário,
2013-01-01 a 2017-12-31 (1826 dias).

**Como obter:** baixar `train.csv` no Kaggle e colocar em `data/raw/train.csv`
(não é versionado).

## Desenho experimental

| Item | Definição |
|---|---|
| Lookback | 364 dias (52 semanas) |
| Horizonte | 90 dias |
| Treino | até 2016-06-30 |
| Validação | 2016-07-01 a 2016-12-31 |
| Teste | 2017, 4 janelas de 90 dias sem sobreposição a partir de 2017-01-01 (origens: 01/01, 01/04, 30/06, 28/09) |
| Métricas | MAE, RMSE, sMAPE, MASE (sazonalidade 7, escala no treino de cada série); média sobre as séries |
| Baselines | naive sazonal semanal; naive sazonal anual (valor de 364 dias antes) |

Notas:
- 4 x 90 = 360 dias, então a última janela termina em **2017-12-26**; os 5 últimos dias
  do dataset não entram no teste.
- Janelas de treino/validação: o horizonte previsto fica dentro do intervalo do split; o
  lookback pode usar histórico anterior. Amostras channel-independent (uma por série e origem).
- **Lookback = 364.** Os dados são diários, e 364 dias = 52 semanas: o modelo enxerga o mesmo dia da semana do ano anterior, o que torna justa a comparação com o baseline sazonal anual (mesma informação disponível). Além disso, 364 é divisível por 7, 14 e 28, então os patches da Fase 2 se alinham a semanas. O 336 do paper vem de datasets horários e não se aplica aqui.
- Seed fixa: 42.

## Como rodar

```bash
uv sync
uv run python -m src.eda         # figuras em results/figures/
uv run python -m src.baselines   # results/tables/baselines.csv
```

## Estrutura

```
data/raw/        train.csv (não versionado)
data/processed/  derivados (não versionado)
src/config.py    constantes e caminhos
src/data.py      carga, origens de teste, janelas
src/metrics.py   mae, rmse, smape, mase
src/baselines.py naive semanal e anual
src/eda.py       figuras e estatísticas
results/         figures/ e tables/
```

## Status

- [x] Setup (uv, git), config, carga de dados, métricas
- [x] Baselines naive sazonais
- [x] EDA
- [ ] Implementação do MTST (Fase 2)
- [ ] Treino global
- [ ] Treino local
- [ ] Comparação global vs. local nas 4 janelas de teste

## Referências

- Zhang, Y.; Ma, L.; Pal, S.; Zhang, Y.; Coates, M. *Multi-resolution Time-Series Transformer for
  Long-term Forecasting*. AISTATS 2024 (PMLR 238).
- Código oficial: https://github.com/networkslab/MTST
