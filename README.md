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
  batch 128, janelas de treino com stride 7. Épocas: ver v1 e v2 em Resultados (v2 atual: 15
  épocas fixas, sem early stopping, melhor checkpoint pela validação). Na avaliação, as
  previsões são desnormalizadas, truncadas em >= 0 e as métricas calculadas na escala original.
- **Dupla normalização (observação):** as janelas passam por duas normalizações. A padronização
  por série (média e desvio do treino) é feita por nós e o RevIN, ligado no forward em treino e
  avaliação, normaliza de novo cada janela de entrada (média e desvio da própria janela, sem
  parâmetros afins). Como o RevIN é invariante a deslocamento e escala da entrada, ele anula a
  padronização na entrada do modelo; o efeito dela fica só na **loss**, onde iguala o peso das
  séries (a saída é desnormalizada pelo RevIN para a escala da série padronizada). Não foi
  testado o treino sem a padronização.
- **Modos:** `--modo global` (500 séries), `--modo local --loja N`, `--modo local_todas`
  (um modelo por loja, métricas sobre as 500 séries).

## Resultados (média das 4 janelas de teste)

Duas configurações de treino (o resto da configuração é igual):

- **v1:** OneCycleLR planejado para 30 épocas + early stopping (paciência 4). **Problema do
  agendador:** o pico do lr cai na época ~9 e o early stopping parou o treino por volta da
  época 6 a 12, antes da fase de decaimento (nos locais, o melhor ponto foi a época 2, com o lr
  ainda subindo).
- **v2:** OneCycleLR com 15 épocas fixas (max_lr 5e-4, pct_start 0,3), sem early stopping; só
  o melhor checkpoint pela loss de validação é guardado e usado no teste.

| modelo | MAE | RMSE | sMAPE | MASE |
|---|---|---|---|---|
| naive semanal | 11.421 | 14.011 | 21.589 | 1.397 |
| **naive anual** | **8.564** | **10.759** | **17.115** | **1.073** |
| MTST global v1 | 10.614 | 12.934 | 19.185 | 1.281 |
| MTST global v2 | 10.595 | 12.943 | 19.131 | 1.279 |
| MTST local v1 (1 por loja) | 9.886 | 12.229 | 18.015 | 1.197 |
| MTST local v2 (1 por loja) | 10.627 | 13.022 | 19.300 | 1.285 |

Tabela por janela: `results/tables/comparacao.csv` (arquivos `*_v1.csv` = v1). Figuras:
`results/figures/previsoes_exemplo.png` (v2) e `curvas_treino.png` (v2, com o lr).

- **Nenhuma configuração bate o naive sazonal anual** (MASE 1,07).
- **Global:** v1 e v2 praticamente iguais (1,281 vs 1,279); o decaimento do lr não mudou o resultado.
- **Local:** a v2 ficou **pior** que a v1 (1,285 vs 1,197), mesmo com o ciclo completo: treinar
  mais (melhor época 10 a 15 contra 2) melhora a loss de treino, mas não a generalização. A v1,
  que parou cedo, funcionou como uma regularização. Nas lojas 4 e 10 o melhor ponto da v2 foi a
  época 1. Com a v2, global e local ficam equivalentes (1,279 vs 1,285).
- **Melhor época (v1 / v2):** global 8 / 8; locais 2 / 10, 13, 15, 1, 15, 13, 15, 14, 13, 1 (lojas 1 a 10).
- **Tempo de treino (v1 / v2):** global 44,9 / 57,5 min; 10 locais 23,2 / 66,7 min; total 68 / 124 min.
- Gap treino x validação grande em todos (ex.: global 0,32 vs 0,86 na loss normalizada), sugerindo
  overfitting ou deslocamento de nível entre treino e validação (hipótese, não verificada).
  Uma seed, sem ajuste de hiperparâmetros.

## Como rodar

```bash
uv sync
uv run python -m src.eda         # figuras em results/figures/
uv run python -m src.baselines   # results/tables/baselines.csv
uv run python -m src.train --modo global   # ou: --modo local --loja 3 | --modo local_todas
uv run python -m src.comparar    # comparacao.csv e figuras (após treinar global e local_todas)
# teste rápido: --epocas 1 --amostra 0.1
```

## Estrutura

```
data/raw/        train.csv (não versionado)
data/processed/  derivados (não versionado)
src/config.py    constantes e caminhos
src/data.py      carga, origens de teste, janelas
src/metrics.py   mae, rmse, smape, mase
src/baselines.py naive semanal e anual
src/train.py     treino e avaliação do MTST
src/comparar.py  tabela de comparação e figuras
src/mtst_oficial/ modelo oficial vendorizado
src/eda.py       figuras e estatísticas
results/         figures/ e tables/
```

## Status

- [x] Setup (uv, git), config, carga de dados, métricas
- [x] Baselines naive sazonais
- [x] EDA
- [x] MTST oficial vendorizado, config diária e `train.py` (smoke test OK)
- [x] Treino global
- [x] Treino local (um modelo por loja)
- [x] Comparação global vs. local nas 4 janelas de teste (v1 e v2)
- [x] Correção do agendador (v2): o decaimento do lr não melhorou o resultado
- [ ] Próximos: regularização (dropout, weight decay, modelo menor), corrigir o deslocamento de
  nível, mais seeds; avaliar o MTST local com early stopping por loja

## Referências

- Zhang, Y.; Ma, L.; Pal, S.; Zhang, Y.; Coates, M. *Multi-resolution Time-Series Transformer for
  Long-term Forecasting*. AISTATS 2024 (PMLR 238).
- Código oficial: https://github.com/networkslab/MTST
