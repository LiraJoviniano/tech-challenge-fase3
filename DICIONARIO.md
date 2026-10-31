# Dicionário de Dados

**Projeto:** Predição de Alfabetização e Inteligência Analítica para Educação no Brasil (Tech Challenge — Fase 3)
**Escopo:** a Bronze própria da Fase 3, os insumos lidos da Fase 2 e a Gold `features_aluno`, consumida pelos modelos.

> **Fontes oficiais:**
> [Indicador Criança Alfabetizada](https://basedosdados.org/dataset/073a39d4-89cf-4068-b1e8-34ed0d9c0b72) (INEP, via Fase 2) ·
> Censo Demográfico 2022 (`basedosdados.br_ibge_censo_2022`) ·
> PIB dos Municípios (`basedosdados.br_ibge_pib`).
> Os códigos herdados da Fase 2 (`rede`, `alfabetizado`, `presenca`) seguem o
> [dicionário da Fase 2](https://github.com/luizafcunha/fiap-ai-scientist-fase-02/blob/main/DICIONARIO.MD).

**Bucket do data lake:** compartilhado com a Fase 2. A Fase 3 escreve apenas sob o prefixo `fase3/`.

| Database (Glue Catalog) | Caminho no S3 | Conteúdo |
|---|---|---|
| `fase3_bronze` | `s3://<bucket>/fase3/bronze/` | Fontes socioeconômicas, catalogadas por crawler |
| `fase3_gold` | `s3://<bucket>/fase3/gold/features_aluno/` | Dataset de modelagem, schema declarado em Terraform |
| `alfabetizacao_silver`, `alfabetizacao_gold` | prefixos da Fase 2 | Somente leitura |

## Legenda de classificação de dado (LGPD)

| Classificação | Significado |
|---|---|
| 🟢 Público | Dado agregado/estatístico, sem risco de identificação individual |
| 🟡 Pseudonimizado | Identificador que não é PII direto, mas não deve ser exposto sem necessidade |
| 🔴 Sensível | Resultado individual de uma criança |

## Convenções

**Unidades.** Colunas com prefixo `mun_pct_` e as taxas do indicador estão em **0 a 100**.
`mun_taxa_alfabetizacao_adulta` é a exceção: está em **fração de 0 a 1**, como vem do Censo.

**Prefixo `mun_`.** Toda coluna com esse prefixo descreve o **município**, não o aluno, e se
repete em todos os alunos da mesma cidade. É a origem da pseudo-replicação tratada com
`GroupKFold` por `id_municipio`.

**Alvo no modelo.** Na Gold, `alfabetizado = True` é a criança alfabetizada. No modelo o alvo
é **invertido** (`alvo_invertido: true` em `src/config/features_modelo.json`): `1` = não
alfabetizado, a classe de risco.

**`rede_codigo`.** Sempre `3` (Municipal) — o recorte casa com as metas municipais.

**`regiao`.** Norte, Nordeste, Sudeste, Sul ou Centro-Oeste.

---

# Camada Bronze (Fase 3)

Dados extraídos do BigQuery sem transformação, gravados com histórico preservado.

**Caminho:** `fase3/bronze/<fonte>_<tabela>/dt_ingestao=AAAA-MM-DD/dados.parquet`

| Mecanismo | Função |
|---|---|
| Partição `dt_ingestao` | Padrão Hive; cada carga coexiste com as anteriores |
| Coluna `_ingerido_em` (TIMESTAMP UTC) | Presente em toda tabela; instante da extração |
| `reports/ingestao/manifesto.jsonl` | Uma linha por execução: linhas, colunas, MB varridos, caminho e consulta |

> A pasta da tabela vem antes da partição, e o nome combina fonte e tabela: sem isso,
> `censo_2022/municipio` e `pib_municipal/municipio` colidiriam no crawler.

## `censo_2022_municipio`

**Origem:** `basedosdados.br_ibge_censo_2022.municipio`
**Volume:** 5.570 linhas · 0,41 MB varridos · **Granularidade:** município

| Coluna | Tipo | Descrição | Faixa observada |
|---|---|---|---|
| `id_municipio` | STRING (7) | Código IBGE | 5.570 valores |
| `sigla_uf` | STRING | UF | 27 valores |
| `populacao` | INT | População residente | 829 a 11.395.160 |
| `domicilios` | INT | Domicílios particulares | 322 a 4.307.665 |
| `area` | INT | Área territorial, km², armazenada como inteiro | 4 a 159.533 |
| `taxa_alfabetizacao_adulta_censo` | DOUBLE | Taxa de alfabetização da população de 15 anos ou mais, **fração 0–1**. Renomeada na extração (origem: `taxa_alfabetizacao`) para não ser confundida com o indicador do INEP | 0,632 a 0,991 |
| `idade_mediana` | DOUBLE | Idade mediana da população, em anos | 15 a 53 |
| `indice_envelhecimento` | DOUBLE | Índice de envelhecimento do IBGE (idosos por 100 jovens) | 5,4 a 277,1 |
| `populacao_indigena` | INT | Pessoas indígenas | 0 a 71.286 |
| `populacao_quilombola` | INT | Pessoas quilombolas | 0 a 15.973 |

## `censo_2022_alfabetizacao_grupo_idade_sexo_raca`

**Origem:** `basedosdados.br_ibge_censo_2022.alfabetizacao_grupo_idade_sexo_raca`
**Volume:** 779.800 linhas · 47,42 MB varridos
**Granularidade:** município × faixa etária × sexo × cor/raça × condição de alfabetização

| Coluna | Tipo | Descrição |
|---|---|---|
| `id_municipio` | STRING (7) | Código IBGE |
| `grupo_idade` | STRING | 7 faixas etárias, a partir de `15 a 19 anos` |
| `sexo` | STRING | `Homens`, `Mulheres` |
| `cor_raca` | STRING | `Branca`, `Preta`, `Parda`, `Amarela`, `Indígena` |
| `alfabetizacao` | STRING | `Alfabetizadas`, `Não alfabetizadas` |
| `populacao` | INT | Pessoas na combinação — **32,1% nulos** (combinações sem contagem publicada) |

> ⚠️ **Ingerida e catalogada, mas não consumida.** Nenhum script, notebook ou o Job da Gold
> lê esta tabela. A Gold usa a taxa de 15 anos ou mais de `censo_2022_municipio`.

## `pib_municipal_municipio`

**Origem:** `basedosdados.br_ibge_pib.municipio`
**Volume:** 16.710 linhas (5.570 municípios × 3 anos) · 6,32 MB varridos
**Granularidade:** município × ano

| Coluna | Tipo | Descrição |
|---|---|---|
| `id_municipio` | STRING (7) | Código IBGE |
| `ano` | INT | 2021, 2022 ou 2023 — o último ano é descoberto em tempo de execução |
| `pib` | INT | PIB a preços correntes, em R$ |
| `va_agropecuaria` | INT | Valor adicionado da agropecuária, R$ |
| `va_industria` | INT | Valor adicionado da indústria, R$ |
| `va_servicos` | INT | Valor adicionado dos serviços (exceto administração pública), R$ |
| `va_adespss` | INT | Valor adicionado da administração, defesa, educação e saúde públicas e seguridade social, R$ |

> ⚠️ **As colunas `va_*` só têm valor em 2021** (66,7% de nulos). O IBGE publica o PIB total
> antes da desagregação por atividade. A Gold usa o PIB de 2023 e a composição de 2021.
>
> ⚠️ **Valor adicionado pode ser negativo.** Faina (GO) tem `va_industria` de −477 milhões
> em 2021, provável ajuste contábil. A Gold anula as proporções em vez de invertê-las.

---

# Insumos da Fase 2 (somente leitura)

Sincronizados por `src/ingestion/sincronizar_fase2.py` e lidos pelo Job da Gold via Glue
Catalog. A Fase 3 não grava nos prefixos da Fase 2. Definições completas no dicionário da Fase 2.

| Tabela | Camada | Colunas lidas | Uso |
|---|---|---|---|
| `fato_aluno` | Silver | `ano`, `id_aluno`, `id_escola`, `id_municipio`, `rede_codigo`, `caderno`, `alfabetizado`, `peso_aluno` | Base: um aluno por linha. Filtros: `ano = 2024`, `aluno_valido`, `rede_codigo = '3'`, `alfabetizado` não nulo |
| `dim_territorio` | Silver | `id_municipio`, `sigla_uf`, `regiao` | Território |
| `features_municipio` | Gold | `taxa_2023`, `total_escolas`, `alunos_por_docente`, `alunos_por_turma`, `pct_matricula_integral`, `indice_infraestrutura`, `pct_matricula_rural`, `pct_matricula_transporte` | Contexto educacional |
| `trajetoria_meta_2030` | Gold | `meta_2024`, `taxa_2023`, `elegivel_meta` | Meta pactuada |
| `fato_escola` | Silver | — | Sincronizada, mas **não usada**: o `id_escola` de `fato_aluno` é sintético e o join retorna zero linhas |

**Colunas deliberadamente fora da Gold:**

| Origem | Colunas | Motivo |
|---|---|---|
| `fato_aluno` | `proficiencia`, `distancia_corte`, `faixa_proximidade` | Definem o alvo por construção (vazamento direto) |
| `features_municipio` | `alvo_taxa_2024`, `alvo_atingiu_meta`, `variacao_anual` | Média do alvo da própria turma, ou derivada de `taxa_2024` (vazamento temporal) |
| `trajetoria_meta_2030` | `taxa_2024`, `atingiu_meta_2024`, `distancia_meta_2030`, `ritmo_necessario`, `indice_trajetoria`, `classificacao_trajetoria` | Todas derivam de `taxa_2024`, o resultado a prever |

---

# Camada Gold

## `features_aluno`

**Database:** `fase3_gold` · **Construída por:** `src/transformation/gold_fase3.py` (Glue Job)
**Granularidade:** aluno avaliado em 2024, rede Municipal · **Chave:** `ano` + `id_aluno`
**Volume:** 1.610.754 linhas · 5.450 municípios · 25 UFs · 36.920 escolas · 35 colunas

| Métrica | Valor |
|---|---:|
| Alvo positivo (`alfabetizado = True`) | 59,4% |
| Linhas do RS (`uf_anomala`) | 61.053 |
| Alunos por município, contando cada município uma vez — média · mediana · máximo | 295,6 · 103 · 43.340 |
| Tamanho do município do aluno típico (mediana contando cada aluno uma vez) | 944 — valor aproximado do Athena (`APPROX_PERCENTILE`), como no README |

A coluna **Modelo** indica o tratamento no pipeline (`src/config/features_modelo.json`):
**log** = mediana + `log1p` + padronização · **padrão** = mediana + padronização ·
**teto** = mediana + teto no percentil 99 do treino + padronização · **one-hot** = mais
frequente + one-hot · **excluída** = fora do modelo, com o motivo.

### Chaves e agrupamento

| Coluna | Tipo | Descrição | Nulos | Modelo | LGPD |
|---|---|---|---:|---|---|
| `ano` | INT | Sempre 2024 | 0% | excluída — constante | 🟢 |
| `id_aluno` | STRING | Identificador sintético da avaliação. Único dentro do ano | 0% | excluída — identificador | 🟡 |
| `id_escola` | STRING | Identificador **sintético**, todos com prefixo `60`. Não é o código INEP e é regenerado a cada ano | 0% | excluída — identificador | 🟡 |
| `id_municipio` | STRING (7) | Código IBGE | 0% | excluída como feature; é a **chave de grupo** da divisão treino/teste | 🟢 |
| `sigla_uf` | STRING | UF. DF e RR ausentes da fonte | 0% | one-hot | 🟢 |
| `regiao` | STRING | Região geográfica | 0% | one-hot | 🟢 |

### Alvo e peso

| Coluna | Tipo | Descrição | Nulos | Modelo | LGPD |
|---|---|---|---:|---|---|
| `alfabetizado` | BOOLEAN | `True` se `proficiencia >= 743` na escala Saeb | 0% | alvo, invertido | 🔴 |
| `peso_aluno` | DOUBLE | Peso amostral da avaliação, de 1,00 a 29,83 | 0% | excluída — descreve o desenho da avaliação, não o aluno | 🟢 |

### Marcações e atributos do aluno

| Coluna | Tipo | Descrição | Nulos | Modelo | LGPD |
|---|---|---|---:|---|---|
| `uf_anomala` | BOOLEAN | `sigla_uf = 'RS'`. Marca a queda atípica do estado em 2024; o RS é marcado, não removido | 0% | excluída — idêntica a `sigla_uf == "RS"` | 🟢 |
| `rede_codigo` | STRING | Sempre `3` | 0% | excluída — constante | 🟢 |
| `caderno` | STRING | Caderno de prova aplicado, 22 valores. Única candidata a feature que varia dentro do município | 0% | excluída — só existe no dia da prova | 🟢 |

### Contexto educacional do município

Herdado de `features_municipio` (Fase 2). Exceto a taxa, as colunas vêm do **Censo Escolar de
2024**, agregado por município sobre escolas da rede Municipal em atividade que ofertam anos
iniciais, ponderado por matrícula.

| Coluna | Tipo | Descrição | Faixa | Nulos | Modelo |
|---|---|---|---|---:|---|
| `mun_taxa_ano_anterior` | DOUBLE | Taxa de alfabetização do município em **2023**, rede Municipal (`taxa_2023`) | 4,35 a 100 | 0,51% | padrão |
| `mun_total_escolas` | INT | Escolas do universo acima | 1 a 759 | 0% | log |
| `mun_alunos_por_docente` | DOUBLE | Matrículas ÷ docentes dos anos iniciais | 3,9 a 120,9 | 0% | teto |
| `mun_alunos_por_turma` | DOUBLE | Matrículas ÷ turmas dos anos iniciais | 4,6 a 109,9 | 0% | teto |
| `mun_pct_integral` | DOUBLE | % de matrículas em tempo integral | 0 a 100 | 0% | padrão |
| `mun_indice_infraestrutura` | DOUBLE | Média simples de seis % de matrícula com: biblioteca, laboratório de informática, banda larga, água adequada, energia pública e esgoto adequado | 11,4 a 100 | 0% | padrão |
| `mun_pct_rural` | DOUBLE | Matrículas de residentes em zona rural ÷ matrículas dos anos iniciais | 0 a **552,6** | 0% | teto |
| `mun_pct_transporte` | DOUBLE | Matrículas que usam transporte público ÷ matrículas dos anos iniciais | 0 a **466,9** | 0% | teto |

> ⚠️ **Quatro razões passam do possível.** Em `mun_pct_rural` e `mun_pct_transporte` o
> numerador conta todas as etapas e o denominador só os anos iniciais — acima de 100% em
> 1.279 e 1.105 municípios. Em `mun_alunos_por_turma` e `mun_alunos_por_docente`, turmas
> multisseriadas não contam como anos iniciais (Jordão-AC: 109,9 alunos por turma, contra
> mediana de 20,3). Funcionam como índice, não como percentual. O teto no percentil 99 é
> aprendido só no treino.

### Meta pactuada

| Coluna | Tipo | Descrição | Faixa | Nulos | Modelo |
|---|---|---|---|---:|---|
| `mun_meta_ano_alvo` | DOUBLE | Meta de 2024 pactuada pelo município (`meta_2024`) | 7,94 a 80 | 2,62% | excluída — redundante: é `taxa − distância` |
| `mun_distancia_meta_anterior` | DOUBLE | `taxa_2023 − meta_2024`: quanto faltava no fim de 2023. Negativo = abaixo da meta | −7,22 a 32,01 | 2,62% | padrão |
| `mun_elegivel_meta` | BOOLEAN (nullable) | Município com meta publicada e resultado comparável | — | 0,001% | binária, nulo vira `0` |

### Contexto socioeconômico

Censo Demográfico 2022 e PIB municipal. Nenhuma contagem absoluta entra no modelo: tudo é
proporção, taxa ou densidade.

| Coluna | Tipo | Descrição | Cálculo | Faixa | Nulos | Modelo |
|---|---|---|---|---|---:|---|
| `mun_populacao` | INT | População residente, 2022 | direto | 829 a 11.395.160 | 0% | log |
| `mun_densidade` | DOUBLE | Habitantes por km² | `populacao ÷ area` | 0,15 a 13.666 | 0% | log |
| `mun_moradores_por_domicilio` | DOUBLE | Adensamento domiciliar | `populacao ÷ domicilios` | 2,29 a 5,69 | 0% | padrão |
| `mun_taxa_alfabetizacao_adulta` | DOUBLE | Alfabetização da população de 15 anos ou mais, **fração 0–1** | direto | 0,653 a 0,991 | 0% | padrão |
| `mun_idade_mediana` | DOUBLE | Idade mediana, em anos | direto | 17 a 51 | 0% | excluída — correlação acima de 0,9 |
| `mun_indice_envelhecimento` | DOUBLE | Idosos por 100 jovens | direto | 6,6 a 246,0 | 0% | padrão |
| `mun_pct_indigena` | DOUBLE | % da população indígena | `100 × indígena ÷ população` | 0 a 96,3 | 0% | log |
| `mun_pct_quilombola` | DOUBLE | % da população quilombola | `100 × quilombola ÷ população` | 0 a 84,5 | 0% | log |
| `mun_pib_per_capita` | DOUBLE | PIB por habitante, R$ | `pib 2023 ÷ população 2022` | 7.708 a 724.813 | 0% | log |
| `mun_pct_va_agropecuaria` | DOUBLE | Participação da agropecuária no valor adicionado, 2021 | `100 × va ÷ soma dos quatro va` | 0 a 93,5 | 0,006% | padrão |
| `mun_pct_va_industria` | DOUBLE | Participação da indústria | idem | 0,5 a 93,3 | 0,006% | padrão |
| `mun_pct_va_servicos` | DOUBLE | Participação dos serviços | idem | 2,0 a 87,7 | 0,006% | padrão |
| `mun_pct_va_administracao_publica` | DOUBLE | Participação da administração pública | idem | 0,8 a 76,3 | 0,006% | padrão |

> As quatro `mun_pct_va_*` são nulas quando algum valor adicionado é negativo ou a soma não
> é positiva — hoje, só Faina (GO). Somam 100 nos demais municípios.

### Resumo do uso no modelo

| Situação | Qtde | Colunas |
|---|---:|---|
| Entram no modelo | 24 | 6 log, 11 padrão, 4 teto, 2 one-hot, 1 binária |
| Alvo | 1 | `alfabetizado` |
| Chave de grupo | 1 | `id_municipio` |
| Excluídas | 9 | `ano`, `rede_codigo`, `id_aluno`, `id_escola`, `peso_aluno`, `caderno`, `uf_anomala`, `mun_meta_ano_alvo`, `mun_idade_mediana` |

### Validação aplicada pelo Job antes de gravar

| Regra | Tipo |
|---|---|
| Chave `(ano, id_aluno)` duplicada | Bloqueante |
| `id_municipio` fora de 7 dígitos | Bloqueante |
| `alfabetizado` nulo | Bloqueante |
| Coluna de vazamento presente | Bloqueante |
| Coluna 100% nula | Bloqueante |
| `mun_pct_*` fora de 0–100 | Alerta — dispara hoje por `mun_pct_rural` e `mun_pct_transporte` |

---

# Saídas da modelagem

Arquivos versionados em `reports/modelagem/`, consumidos pelo README e pela comparação entre modelos.

| Arquivo | Granularidade | Colunas principais |
|---|---|---|
| `probas_teste_<modelo>.parquet` | aluno do teste (290.331) | `proba`: probabilidade de risco. Sem `id_aluno`: o índice é o número da linha na Gold completa, o mesmo de `y_teste` |
| `comparacao_teste.csv` | modelo | `roc_auc`, `pr_auc`, `recall`, `precisao`, `f1`, `alertas (% dos alunos)`, `limiar` |
| `calibracao_teste.csv` | modelo | `brier`, `brier_skill`, `ece_pontos`, `risco_medio_previsto_pct`, `risco_observado_pct`, `alunos` |
| `clusters_municipios.csv` | município (5.450) | `id_municipio`, `sigla_uf`, `regiao`, `grupo` (1 a 4), `alunos`, `taxa_alfabetizacao` (0–100) |
| `metas_municipios_risco.csv` | município do backtest (889) | `taxa_prevista`, `taxa_observada`, `meta`, `distancia_prevista`, `distancia_observada`, `previsto_abaixo`, `observado_abaixo` |
| `interpretacao_uf.csv` | UF | SHAP médio da UF em log-odds de risco, `com_historico` e `sem_historico` |
| `interpretacao_ranking.csv` | variável | `importancia_com_historico` e `importancia_sem_historico` (\|SHAP\| médio); `direcao_com_historico` e `direcao_sem_historico` (positivo = maior valor, mais risco) |

---

## Governança do dado individual

`features_aluno` está no grão da criança e carrega o resultado individual (`alfabetizado`).
As contenções:

| Contenção | Situação |
|---|---|
| Sem nota | `proficiencia` e derivadas não existem na Gold |
| Identificadores sintéticos | `id_aluno` e `id_escola` são atribuídos pela avaliação; `id_escola` é regenerado a cada ano e não liga a escola real |
| Uso agregado | O modelo estima o risco do contexto. O uso legítimo é priorizar municípios e redes, não rotular crianças |

> **Diferença em relação à Fase 2.** Lá a regra era "nenhum Parquet vai para o
> repositório". Aqui, `data/sample/features_aluno/` é versionado de propósito, com as
> 1.610.754 linhas: torna os notebooks executáveis sem depender do S3, que expira com o
> laboratório (README, seção 14).

---

## Como conferir

```bash
python -c "import pandas as pd; df = pd.read_parquet('data/sample/features_aluno'); print(df.shape); print(df.dtypes); print(df.isna().mean().mul(100).round(3).sort_values(ascending=False).head(10))"
```
