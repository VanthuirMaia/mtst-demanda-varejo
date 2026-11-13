# MTST em previsão de demanda de varejo (M5)

Projeto da disciplina Previsão de Séries Temporais (mestrado).

## Pergunta central

**O ganho do MTST sobre o PatchTST, mostrado nos benchmarks do artigo, se mantém em previsão de
demanda de varejo?**

O *Multi-resolution Time-Series Transformer* (Zhang et al., AISTATS 2024) divide a série em
patches de vários tamanhos (um ramo do Transformer por tamanho) e usa *positional encoding*
relativo. O PatchTST usa um único tamanho de patch.

- **Parte 1 – Replicação (ETTh1):** reproduz o experimento do artigo para validar a nossa
  configuração do MTST (seção *Replicação*; roda no Colab).
- **Parte 2 – Aplicação (M5):** MTST e PatchTST **globais** (um modelo para todas as séries,
  channel-independent) no M5, com **naive sazonal** e **ARIMA por série** como referências clássicas.
- **Store Item Demand Forecasting:** *experimento preliminar* (500 séries sintéticas, 2013-2017),
  fora do fluxo principal. O código continua executável com `DATASET=store_item`; resultados
  preliminares em `results/` sem prefixo `m5_`.

## Dataset de aplicação: M5 (Walmart)

`sales_train_evaluation.csv` e `calendar.csv` (Kaggle, *M5 Forecasting - Accuracy*) em
`data/raw/m5/` (não versionados). Vendas agregadas por **loja x departamento**: **70 séries**
(10 lojas x 7 departamentos), colunas `"{store}_{dept}"`, 1941 dias, 2011-01-29 a 2016-05-22, sem NaN.

**Características do dataset (não tratadas):**
- **Natal:** nos dias 25/12 de 2011 a 2015 a venda total cai a ~0 (11 a 20 unidades, contra ~34 mil
  de média): lojas fechadas. Não há outros dias com total próximo de zero e nenhuma loja inteira
  ficou zerada em outro dia. Esses dias ficam no treino e, no teste, o 25/12/2015 cai na janela 3.
- Séries de escalas muito diferentes (média diária de 12 a 2.913 unidades) e raros zeros por
  série (0,27% dos dias em média; máx. 1,08% em `WI_1_HOBBIES_2`).
- Perfil mensal (figura) enviesado pelos anos parciais (2011 começa em 29/01; 2016 termina em 22/05).

### M5 vs. Store Item (`results/tables/m5_vs_store_item_eda.csv`)

| | M5 (loja x depto) | Store Item (loja x item) |
|---|---|---|
| Séries x dias | 70 x 1941 | 500 x 1826 |
| Escala média diária (min / max) | 493 (12 / 2.913) | 52 (13 / 113) |
| Razão max/min da escala | 244 | 8,9 |
| CV médio (min / max) | 0,36 (0,21 / 0,80) | 0,29 (0,26 / 0,37) |
| % de dias zero (média / máx.) | 0,27 / 1,08 | 0,00 / 0,05 |

O Store Item é bem mais homogêneo (escalas e CV em faixas estreitas, praticamente sem zeros),
coerente com dados sintéticos; o M5 (real) tem heterogeneidade de escala e de variabilidade bem
maiores e eventos como o Natal.

## Desenho experimental

| Item | Definição |
|---|---|
| Lookback / horizonte | 364 / 90 dias (os dois datasets) |
| Splits (M5) | **calculados a partir das datas**: teste = últimos 360 dias em 4 janelas de 90; validação = 180 dias antes do teste; treino = o resto |
| Treino | 2011-01-29 a 2014-11-29 (1401 dias) |
| Validação | 2014-11-30 a 2015-05-28 (180 dias) |
| Teste | 2015-05-29 a 2016-05-22; origens 05-29, 08-27, 11-25, 02-23 |
| Métricas (escala original) | MAE, RMSE, sMAPE, MASE (sazonalidade 7, escala no treino); média sobre séries |
| Métricas (normalizada) | MSE e MAE após padronizar cada série com média e desvio do treino (protocolo do artigo) |
| Baselines | naive sazonal semanal e anual (364 d); AutoARIMA (statsforecast, sazonalidade 7) por série |

- **ARIMA com origem móvel:** para cada janela, ajustado em todo o histórico anterior à origem
  (treino + validação + janelas de teste anteriores), previsão de 90 dias clipada em >= 0.
- **ARIMA x Transformers:** o ARIMA é **reajustado a cada origem** (histórico completo até a
  origem). Os Transformers são **treinados uma vez** (treino + validação de seleção) e só usam o
  histórico recente (os 364 dias anteriores) como entrada, sem reajuste de pesos nas janelas de teste.
- Janelas de treino/validação: o horizonte fica dentro do intervalo do split; o lookback pode usar
  histórico anterior. Amostras channel-independent. Seed 42.
- **Lookback = 364:** 52 semanas, o mesmo dia da semana do ano anterior (justo vs. naive anual),
  divisível por 7, 14 e 28. O 336 do artigo vem de dados horários.

## Implementação do MTST

O modelo é o **código oficial** (https://github.com/networkslab/MTST, commit `d52b6b5`),
vendorizado em `src/mtst_oficial/`. Arquivos copiados: `models/MTST.py`,
`layers/MTST_backbone.py`, `layers/PatchTST_layers.py`, `layers/RevIN.py`,
`layers/rel_pe.py`. Única alteração: imports (`layers.*` -> `src.mtst_oficial.*`); cada
arquivo tem cabeçalho com origem e alterações. Dependências extras: `yacs` e `einops`.

- **Licença:** o repositório oficial **não tem arquivo LICENSE** (nem menção no README).
  O código é derivado do PatchTST. Uso aqui é acadêmico, com atribuição.
- **Construção:** o construtor recebe um objeto `configs` (argparse); montamos com
  `types.SimpleNamespace(**config.MTST)`, sem reescrever o modelo.
- **Configuração para dados diários (`config.MTST`):** lookback 364, horizonte 90, `enc_in=1`;
  3 ramos com patch = stride = 7, 14 e 28 dias (52, 26 e 13 tokens); 2 camadas, d_model 64,
  4 cabeças, d_ff 128, dropout 0,2, RevIN e RPE (`rel_sin`) ligados.
- **Divergências em relação aos scripts oficiais** (motivo: custo em CPU e dados diários):
  patches **sem sobreposição** (stride = patch; o oficial usa stride = patch/2, ou seja, ~2x mais
  tokens por ramo), **sem padding** (`padding_patch=None`; o oficial usa `'end'`) e modelo
  **menor** (d_model 64 e 4 cabeças; o oficial usa d_model 128 e 16 cabeças, d_ff 256).
- **Treino (`src/train.py`):** cada série é padronizada com média e desvio do período de
  treino (como o pipeline oficial); loss e early stopping na escala normalizada. RevIN continua
  ligado. Adam com OneCycleLR (max_lr 5e-4, pct_start 0,3, como o lradj `TST` do oficial),
  batch 128, janelas de treino com stride 7. 15 épocas fixas, sem early stopping, melhor
  checkpoint pela validação. Na avaliação, as
  previsões são desnormalizadas, truncadas em >= 0 e as métricas calculadas na escala original.
- **Dupla normalização (observação):** as janelas passam por duas normalizações. A padronização
  por série (média e desvio do treino) é feita por nós e o RevIN, ligado no forward em treino e
  avaliação, normaliza de novo cada janela de entrada (média e desvio da própria janela, sem
  parâmetros afins). Como o RevIN é invariante a deslocamento e escala da entrada, ele anula a
  padronização na entrada do modelo; o efeito dela fica só na **loss**, onde iguala o peso das
  séries (a saída é desnormalizada pelo RevIN para a escala da série padronizada). Não foi
  testado o treino sem a padronização.
- **Treino global:** `uv run python -m src.train` (todas as séries do dataset configurado).

## Seleção de parâmetros

`src/selecao.py`: busca em grade usando **só treino e validação** (o teste não é tocado). Grades
em `src/config.py` (`GRADE`), patches sem sobreposição (stride = patch), `d_ff = 2*d_model`, 4 cabeças,
2 camadas, RevIN (e RPE no MTST) ligados; demais parâmetros como na configuração anterior.

| Modelo | patches | d_model | dropout | configs |
|---|---|---|---|---|
| MTST | (7,14,28) ou (7,28,91) | 32, 64 | 0,2; 0,4 | 8 |
| PatchTST | 7, 14 ou 28 | 32, 64 | 0,2; 0,4 | 6 |

- **Critério:** MSE de validação na escala normalizada (melhor época de cada config).
- **Protocolo (igual para todos):** 15 épocas fixas, Adam com OneCycleLR (max_lr 5e-4, pct_start 0,3),
  batch 128, seed 42, melhor checkpoint pela validação. Janelas de treino com **stride 1**
  (66.360 amostras = 948 origens x 70 séries); validação com stride 7 (910 amostras).
- **Saída:** `results/tables/m5_selecao.csv` (modelo, config, `val_mse_norm`, melhor época, nº de
  parâmetros, tempo). **Retomável:** configs já no CSV são puladas.
- **Treino final (`src/final.py`):** a melhor config de cada modelo é treinada com seeds 42, 43 e 44 e
  avaliada nas 4 janelas de teste (previsão desnormalizada e clipada em >= 0). Saídas: `m5_final.csv`
  (por seed), `m5_comparacao.csv` (naive, ARIMA, PatchTST, MTST; média e desvio amostral entre seeds
  nos Transformers, colunas `<métrica>_dp`), `m5_previsoes_exemplo.png` e `m5_curvas_treino.png` (seed 42).
- **Colab:** `notebooks/m5_colab.ipynb` (GPU; upload de `m5_store_dept.csv` e `pred_m5_arima.npy`).
  Smoke test em CPU (1 época, stride 1): ~129 s/época no MTST e ~76 s no PatchTST;
  `--smoke` grava em `results/smoke/` (ignorado pelo git).

## PatchTST

Código oficial do PatchTST incluído no repositório do MTST (`models/PatchTST.py` e
`layers/PatchTST_backbone.py`, commit `d52b6b5`), vendorizado em `src/patchtst_oficial/` com
cabeçalho de atribuição; só mudam os imports (reaproveita `PatchTST_layers` e `RevIN` de
`src/mtst_oficial/`, idênticos aos do MTST). Verificado por instanciação e forward.

## Resultados (M5, média das 4 janelas de teste)

Baselines (`results/tables/m5_baselines.csv`; MTST e PatchTST ainda não treinados):

| modelo | MAE | RMSE | sMAPE | MASE | MSE norm. | MAE norm. |
|---|---|---|---|---|---|---|
| naive semanal | 95,77 | 122,77 | 21,55 | 1,334 | 1,520 | 0,834 |
| naive anual | 112,59 | 139,72 | 25,83 | 1,641 | 3,086 | 1,009 |
| **ARIMA por série** | **85,71** | **111,00** | **19,18** | **1,202** | **1,283** | **0,755** |

O ARIMA (AutoARIMA, sazonalidade 7) é o melhor baseline em todas as métricas; tempo de ajuste e
previsão: ~34 min nas 4 janelas (70 séries, CPU 8 threads, ~8,5 min por janela).

## Como rodar

```bash
uv sync
uv run python -m src.m5          # valida o M5 e lista os dias com venda ~0
uv run python -m src.eda         # figuras m5_*, resumo e comparação com o Store Item
uv run python -m src.baselines   # naive + ARIMA (~34 min); --sem-arima para só os naive
uv run python -m src.selecao     # grade de parâmetros (retomável; --smoke para teste)
uv run python -m src.final       # treino final, teste, m5_comparacao.csv e figuras
uv run python -m src.train --modelo mtst_global   # execução avulsa (--epocas 1 --amostra 0.1)
# Store Item (preliminar): DATASET=store_item uv run python -m src.<modulo>
```

## Estrutura

```
data/raw/m5/     sales_train_evaluation.csv, calendar.csv (não versionados); train.csv (Store Item)
src/config.py    DATASET, splits, parâmetros
src/m5.py        leitura e validação do M5 (70 séries)
src/data.py      load_data(), origens de teste, janelas
src/metrics.py   mae, rmse, smape, mase, mse_norm, mae_norm
src/baselines.py naive sazonal e ARIMA
src/selecao.py   seleção de parâmetros em grade
src/final.py     treino final (3 seeds), teste e comparação
src/train.py     treino e avaliação do MTST global
src/comparar.py  figuras de comparação
src/mtst_oficial/, src/patchtst_oficial/  modelos oficiais vendorizados
src/eda.py       figuras e estatísticas
results/         figures/ e tables/ (prefixo m5_ = M5); store_item/ = experimento preliminar
```

## Status

- [x] Replicação ETTh1 preparada (Colab)
- [x] Config dataset-agnóstica, dados M5, métricas normalizadas, EDA
- [x] Baselines no M5 (naive e ARIMA)
- [x] PatchTST oficial vendorizado
- [x] Seleção de parâmetros (`selecao.py`), treino final (`final.py`) e notebook Colab prontos; smoke test em CPU OK
- [ ] Rodar seleção e treino final no Colab; analisar a comparação

## Replicação (ETTh1)

**Objetivo:** validar a nossa configuração do MTST reproduzindo o experimento do artigo
(Tabela 2, ETTh1, multivariado, L=336, horizontes 96/192/336/720), independente do varejo.

**O que reproduz do código oficial** (`etth1.sh` + defaults de `run_longExp.py`, commit `d52b6b5`):
- Dados: `Dataset_ETT_hour` (`src/ett.py`): split 12/4/4 meses, sobreposição de `seq_len` em val/teste,
  `StandardScaler` ajustado só no treino, 7 variáveis. Tamanhos (T=96): 8209 / 2785 / 2785 janelas.
- Modelo: 2 camadas, 2 ramos (patch 8 e 16, stride 4 e 8), `padding_patch='end'`, `d_model=128`,
  16 cabeças, `d_ff=256`, dropout 0,3 / fc 0,3 / head 0,1, `rel_sin`, RevIN. Mesmo código do modelo.
- Treino: Adam, MSE, lr 1e-4, `lradj=type3`, batch 256, até 100 épocas, early stopping (paciência 20), seed 2021.
- Métricas: MSE e MAE na escala normalizada, janelas completas de teste (`drop_last`).
- Números do artigo em `results/tables/etth1_artigo.csv`.

Comportamentos do oficial **reproduzidos de propósito** (não são escolha nossa):
- **OneCycleLR nunca avançado:** o `etth1.sh` não passa `--lradj`, então vale `type3` e o `OneCycleLR` é só construído.
  A construção deixa lr=4e-6 e `beta1=0,95` no Adam até o fim da época 1; depois o `type3` põe lr=1e-4 (e `beta1` segue 0,95).
- **`drop_last=True` no teste:** a última janela incompleta é descartada (2560 de 2785 janelas no h=96, batch 256).

Diferenças sem efeito: `StandardScaler` em numpy (sem sklearn); marcas de tempo não geradas (o MTST não as usa);
o teste não é avaliado a cada época (no oficial a loss de teste é só impressa).

```bash
uv run python -m src.replicacao --horizonte 96                       # treino completo
uv run python -m src.replicacao --horizonte 96 --epocas 1 --amostra 0.1   # smoke test
```

Saídas: `results/tables/etth1_resultados.csv` (uma linha por horizonte: MSE, MAE, melhor época, épocas, tempo,
device e diferença abs./relativa vs. artigo) e `etth1_h{H}_log.csv` (curvas por época). Usa GPU se houver, senão CPU
(~13 min/época em CPU de 8 threads; execuções com `--epocas`/`--amostra` são smoke e não entram na tabela de resultados).

### Rodar no Google Colab (GPU)
1. No GitHub, crie um *fine-grained token* com acesso de leitura (*Contents: read*) a este repositório.
2. No Colab, abra `notebooks/replicacao_colab.ipynb` (File > Upload ou via GitHub).
3. Painel de Secrets (ícone de chave): crie `GITHUB_TOKEN` com o token e ative *Notebook access*.
4. *Runtime > Change runtime type > GPU* (T4 serve).
5. *Run all*: clona o repo, instala `einops`/`yacs`, baixa o ETTh1, confere a GPU, roda os 4 horizontes e baixa
   `etth1_resultados.zip` com tabelas e logs. O notebook usa o que estiver no GitHub: faça o push antes.

## Referências

- Zhang, Y.; Ma, L.; Pal, S.; Zhang, Y.; Coates, M. *Multi-resolution Time-Series Transformer for
  Long-term Forecasting*. AISTATS 2024 (PMLR 238).
- Código oficial: https://github.com/networkslab/MTST
