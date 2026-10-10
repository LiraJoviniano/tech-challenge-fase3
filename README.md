# tech-challenge-fase3

FIAP AI Scientist — Fase 3: predição de alfabetização e inteligência analítica para educação no Brasil.

![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![Fontes](https://img.shields.io/badge/fontes-INEP%20%7C%20IBGE-4285F4)
![License](https://img.shields.io/badge/license-MIT-green)

Modelo supervisionado para prever, no grão do estudante, se uma criança será
considerada alfabetizada ao final do 2º ano do Ensino Fundamental — e identificar os
fatores que mais influenciam esse resultado.

Consome a camada Gold construída no [Tech Challenge da Fase 2](https://github.com/luizafcunha/fiap-ai-scientist-fase-02)
e a enriquece com contexto socioeconômico municipal do Censo Demográfico 2022 e do PIB.

> **Legenda de status:** ✅ concluído · 🚧 em andamento · ⏳ planejado
> **Entrega:** 3 de novembro de 2026

---

## Sumário

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo analítico](#2-objetivo-analítico)
3. [Descrição da base utilizada](#3-descrição-da-base-utilizada)
4. [Arquitetura e fronteira com a Fase 2](#4-arquitetura-e-fronteira-com-a-fase-2)
5. [Etapas de modelagem](#5-etapas-de-modelagem)
6. [Escolha do algoritmo](#6-escolha-do-algoritmo)
7. [Métricas de avaliação](#7-métricas-de-avaliação)
8. [Interpretação dos resultados](#8-interpretação-dos-resultados)
9. [Insights encontrados](#9-insights-encontrados)
10. [Limitações do projeto](#10-limitações-do-projeto)
11. [Aplicação prática para políticas públicas](#11-aplicação-prática-para-políticas-públicas)
12. [Possíveis evoluções futuras](#12-possíveis-evoluções-futuras)
13. [Como executar](#13-como-executar)
14. [Estrutura do repositório](#14-estrutura-do-repositório)
15. [Fluxo de trabalho Git](#15-fluxo-de-trabalho-git)
16. [Evidências de execução](#16-evidências-de-execução)
17. [Roadmap e status](#17-roadmap-e-status)
18. [Licença](#18-licença)

---

## 1. Contexto do problema

O Brasil assumiu o compromisso de alfabetizar **todas as crianças até o final do 2º ano
do Ensino Fundamental até 2030**. A medição é feita por avaliação padronizada do INEP, e
considera-se alfabetizado o estudante que atinge **743 pontos na escala Saeb** de Língua
Portuguesa.

A Fase 2 construiu a pipeline que integra resultados, metas e contexto escolar, e
respondeu **o que aconteceu**: 53,3% dos municípios atingiram a meta de 2024 na rede
municipal, e 1.917 retrocederam entre 2023 e 2024.

Descrever o passado, porém, não muda o futuro. Um gestor que recebe o resultado da
avaliação já perdeu o ano letivo — a criança foi avaliada, o diagnóstico chegou depois.
**A lacuna que esta fase ataca é a antecipação:** identificar risco antes da avaliação,
quando ainda há tempo de intervir.

---

## 2. Objetivo analítico

Classificação binária no grão do aluno, prevendo `alfabetizado` (1 se `proficiencia >=
743` na escala Saeb, 0 caso contrário). O alvo está relativamente balanceado — 59,4% de
positivos —, então acurácia continua informativa, mas acompanhada de precisão e recall.

O modelo responde **quem está em risco de não alfabetizar**, a partir de contexto
disponível antes da avaliação (município, infraestrutura escolar, meta pactuada,
histórico do ano anterior). Ele **não** responde por que um aluno específico não
alfabetizou, nem substitui a avaliação em si — é sinal de risco, não diagnóstico.

Classificação, e não regressão sobre a proficiência, porque a decisão de política pública
é binária (aluno entra ou não em programa de apoio) e o corte de 743 pontos já é o
critério oficial de alfabetização — regredir a proficiência adicionaria complexidade sem
mudar a ação tomada a partir do resultado.

---

## 3. Descrição da base utilizada

### Fontes

| Nível | Origem | Tabela | Grão | Papel |
|---|---|---|---|---|
| Aluno | Fase 2 · Silver | `fato_aluno` | aluno × ano | Alvo e atributos individuais |
| Município | Fase 2 · Silver | `dim_territorio` | município | UF e região |
| Município | Fase 2 · Gold | `features_municipio` | município | Contexto educacional |
| Município | Fase 2 · Gold | `trajetoria_meta_2030` | município | Meta pactuada |
| Município | IBGE · Censo 2022 | `municipio` | município | População, domicílios, área, alfabetização adulta |
| Município | IBGE · PIB | `municipio` | município × ano | Capacidade econômica |

**O alvo vem da Silver; a Gold da Fase 3 é construída sobre ele.** O enunciado pede dados
da camada Gold, e a da Fase 2 é agregada por município — insuficiente para predição
individual. Em vez de contornar isso num script de pré-processamento, a Fase 3 declara
sua **própria camada Gold**, `fase3_gold.features_aluno`, no grão do estudante. A camada
que entrega dado pronto para consumo passa a entregar o que este consumo exige.

### `features_aluno` ✅

Uma linha por estudante avaliado em 2024, com 35 colunas.

| Métrica | Valor |
|---|---:|
| Linhas | 1.610.754 |
| Municípios | 5.450 |
| Colunas | 35 |
| Alvo positivo (`alfabetizado`) | 59,4% |
| Sem contexto municipal | 0 |

**Balanceamento saudável.** Com 59,4% de positivos, acurácia continua informativa — o que
não seria verdade numa base 90/10.

**Nulos parciais em 8 colunas.** Entre 0,001% e 2,6% (piores: `mun_meta_ano_alvo` e
`mun_distancia_meta_anterior`), abaixo do limiar que bloqueia a Gold — tratados por
imputação no pipeline de modelagem, não pela validação do Job.

| Grupo | Colunas |
|---|---|
| Chaves e agrupamento | `ano`, `id_aluno`, `id_escola`, `id_municipio`, `sigla_uf`, `regiao` |
| Alvo | `alfabetizado`, `peso_aluno` |
| Marcações | `uf_anomala` |
| Aluno | `rede_codigo`, `caderno` |
| Contexto educacional do município | `mun_taxa_ano_anterior`, `mun_total_escolas`, `mun_alunos_por_docente`, `mun_alunos_por_turma`, `mun_pct_integral`, `mun_indice_infraestrutura`, `mun_pct_rural`, `mun_pct_transporte` |
| Meta pactuada | `mun_meta_ano_alvo`, `mun_distancia_meta_anterior`, `mun_elegivel_meta` |
| Contexto socioeconômico | `mun_populacao`, `mun_densidade`, `mun_moradores_por_domicilio`, `mun_taxa_alfabetizacao_adulta`, `mun_idade_mediana`, `mun_indice_envelhecimento`, `mun_pct_indigena`, `mun_pct_quilombola`, `mun_pib_per_capita`, `mun_pct_va_agropecuaria`, `mun_pct_va_industria`, `mun_pct_va_servicos`, `mun_pct_va_administracao_publica` |

**Das treze colunas de `trajetoria_meta_2030`, apenas três entraram.** As demais —
`distancia_meta_2030`, `ritmo_necessario`, `indice_trajetoria`, `classificacao_trajetoria`
e `atingiu_meta_2024` — derivam de `taxa_2024`, que é o resultado a prever.

O que sobrevive ao corte é informação disponível antes da avaliação: a meta pactuada, a
elegibilidade do município, e `taxa_2023 − meta_2024`, que mede quanto faltava no fim de
2023 para alcançar a meta do ano seguinte. Essa última capta pressão institucional sobre
a rede — município longe da meta tende a mobilizar esforço.

**A taxa de alfabetização adulta usa 25 anos ou mais.** A faixa da fonte começa em 15,
mas 15 a 24 inclui jovens ainda em formação, cuja alfabetização reflete o presente do
sistema e não o contexto familiar da criança. Como a faixa começa em 15 anos, **nenhuma
criança avaliada aparece na tabela** — não há vazamento.

**Nenhuma contagem absoluta entra no modelo.** População indígena de 200 num município de
5 mil habitantes e de 200 em um de 2 milhões descrevem realidades opostas. Tudo vira
proporção, taxa ou densidade.

### Tratamento de vazamento de dados ✅

Requisito explícito do enunciado, e o item que mais distingue a maturidade do projeto.
Três tipos foram identificados e tratados.

**Vazamento direto — colunas que são o alvo disfarçado.** Na Silver da Fase 2,
`alfabetizado` é definido como `proficiencia >= 743`, verificado em 3.354.661 registros
com zero divergências.

| Coluna | Por quê |
|---|---|
| `proficiencia` | Define o alvo por construção |
| `distancia_corte` | `proficiencia − 743` |
| `faixa_proximidade` | Derivada de `distancia_corte` |

**As três não existem na saída.** São excluídas na construção da Gold, não no consumo —
o que impede o uso acidental. Uma regra bloqueante do Job derruba a execução se alguma
delas aparecer no contrato.

**Vazamento temporal.** `features_municipio` traz `alvo_taxa_2024`, que é a média do alvo
dos alunos daquele município. Usá-la para prever um aluno de 2024 vaza o resultado da
própria turma.

A solução foi restringir o dataset a **alunos de 2024 com contexto municipal de 2023**.
Corta o volume à metade e, em troca, o modelo fica genuinamente preditivo: prevê o ano
corrente com informação do ano anterior, que é o uso que um gestor faria.

**Vazamento de contexto — o mais sutil.** As features municipais se repetem em todos os
alunos do município. Medido em `sql/gold/pseudo_replicacao.sql`:

| Métrica | Valor |
|---|---:|
| Linhas | 1.610.754 |
| Municípios | 5.450 |
| Alunos por município (média) | 295,6 |
| Alunos por município (mediana) | 944 |
| Máximo | 43.340 |

O tamanho efetivo da amostra para variáveis municipais é **5.450**, não 1,6 milhão. Um
`train_test_split` aleatório colocaria centenas de alunos do mesmo município em treino e
teste, e o modelo memorizaria o contexto sem generalizar — vazamento sem nenhuma coluna
proibida envolvida.

A correção é `GroupShuffleSplit` e `GroupKFold` agrupando por `id_municipio`, e ela muda
a pergunta que a métrica responde: de *"acerta sobre alunos de municípios que já
conhece"* para *"acerta sobre municípios que nunca viu"* — que é a pergunta de negócio
real, porque o gestor quer antecipar risco onde ainda não mediu.

---

## 4. Arquitetura e fronteira com a Fase 2

A pipeline da Fase 2 permanece em produção como fonte dos dados tratados. A Fase 3 lê e
não modifica: atua apenas sob o prefixo `fase3/` do mesmo bucket, e os crawlers da fase
anterior têm include paths explícitos que não o alcançam.

| Onde | O que acontece | Repositório |
|---|---|---|
| AWS — Glue, Catalog, S3 | Bronze → Silver → Gold da Fase 2 | Fase 2 |
| Local — Python | Ingestão socioeconômica, 54 MB | Fase 3 |
| AWS — Glue Job | Construção de `features_aluno`, 1,6 milhão de linhas | Fase 3 |
| Local — scikit-learn | Modelagem, avaliação, interpretabilidade | Fase 3 |

**Por que não reconstruir a base da Fase 2.** Refazer a tradução de rede, o unpivot das
metas e a agregação ponderada do Censo Escolar não acrescentaria nada e criaria risco de
divergência: dois números para a mesma coisa, sem saber qual está certo.

**Por que ingestão local e transformação em Glue.** A assimetria é deliberada.
Transformar 54 MB em Spark custaria mais em provisionamento do que a transformação
inteira, e o processamento local itera mais rápido. Já a construção de `features_aluno`
são 3,87 milhões de linhas e quatro joins — volume em que o Spark paga o provisionamento.

### Recursos declarados em Terraform ✅

Um `terraform apply` cria **9 recursos**:

| Recurso | Qtde | Papel |
|---|---:|---|
| `aws_glue_catalog_database` | 2 | `fase3_bronze` e `fase3_gold` |
| `aws_glue_crawler` | 1 | Cataloga as fontes socioeconômicas |
| `aws_glue_job` | 1 | Constrói `features_aluno` — Glue 4.0, 2× G.1X |
| `aws_s3_object` | 1 | Upload do script PySpark |
| `aws_glue_catalog_table` | 1 | `features_aluno`, com schema declarado |
| `aws_glue_workflow` | 1 | Orquestração |
| `aws_glue_trigger` | 2 | Encadeamento condicional |

**Crawler na Bronze, schema declarado na Gold.** Na Bronze o schema vem de quem produziu
o dado, e descobri-lo automaticamente é apropriado. Na Gold é decisão: as colunas de
vazamento simplesmente não existem no contrato.

### Histórico de ingestão preservado ✅

| Mecanismo | Efeito |
|---|---|
| Partição `dt_ingestao=AAAA-MM-DD` | Padrão Hive; cargas coexistem em vez de sobrescrever |
| Coluna `_ingerido_em` | Cada registro sabe quando foi lido |
| Manifesto em `reports/ingestao/manifesto.jsonl` | Volume, custo, caminho e a consulta exata de cada execução |

O manifesto é versionado e sobrevive ao laboratório, ao contrário do S3.

**A pasta da tabela vem antes da partição**, no padrão `<fonte>_<tabela>/dt_ingestao=…`.
O Crawler batiza a tabela pela última pasta antes da partição, então `censo_2022/municipio`
e `pib_municipal/municipio` colidiriam — as duas fontes têm uma tabela chamada
`municipio`. A verificação em `infra/executar_crawler.sh` confere que `dt_ingestao` foi
reconhecida como chave e falha se não for.

### Custo da ingestão ✅

Medido por *dry run* antes de cada execução, sem gerar custo:

| Fonte | Tabela | Linhas | Varredura |
|---|---|---:|---:|
| Censo 2022 | `municipio` | 5.570 | 0,41 MB |
| Censo 2022 | `alfabetizacao_grupo_idade_sexo_raca` | 779.800 | 47,42 MB |
| PIB | `municipio` (2021–2023) | 16.710 | 6,32 MB |
| **Total** | | **802.080** | **54,15 MB** |

O BigQuery cobra por coluna varrida: a seleção explícita de colunas é o que reduz custo, e
`LIMIT` corta o retorno, não a varredura.

### Orquestração ✅

O fluxo se encadeia dentro da AWS por **Glue Workflow**:

```
trigger ON_DEMAND
  └─ crawler da Bronze
      └─ (SUCCEEDED) job da Gold
```

A Gold só é gerada se a catalogação tiver sucesso. Sem o encadeamento, o Job leria um
Catalog desatualizado e produziria dado silenciosamente errado: a partição nova não
estaria registrada, e a leitura traria a carga anterior sem qualquer aviso.

**A ingestão fica fora do fluxo, e isso é limitação declarada.** O Glue Workflow encadeia
apenas crawlers e jobs; os dois scripts de ingestão são Python local. Convertê-los em
Glue Jobs Python Shell traria o ciclo inteiro para dentro do Workflow — foi avaliado, e o
custo seria baixo, 0,0625 DPU. Mas as ingestões somam 54 MB e rodam em segundos
localmente; empacotá-las em Glue para orquestrar duas etapas que executam uma vez por ano
não se justifica.

**Por que `ON_DEMAND` e não agendado.** O Censo Demográfico é decenal, o PIB é anual e a
avaliação do INEP é anual. Agendamento diário dispararia execuções reprocessando o mesmo
dado.

### Validação da Gold ✅

O Job valida a saída **antes de gravar**. Quatro regras bloqueantes derrubam a execução:

| Regra | Motivo |
|---|---|
| Chave `(ano, id_aluno)` duplicada | Grão comprometido |
| `id_municipio` fora de 7 dígitos | Join territorial falharia sem erro |
| Alvo nulo | Linha sem o que prever |
| Coluna de vazamento presente | Redundante com o schema, e de propósito |
| Coluna 100% nula | Feature sem valor desaparece do modelo sem erro |

A última virou bloqueante depois de um episódio concreto: as quatro proporções setoriais
do PIB ficaram 100% nulas em duas execuções, porque o IBGE publica o PIB total antes da
desagregação por atividade. O Job passou a escolher o ano mais recente em que cada bloco
de dados existe — PIB total de 2023, composição setorial de 2021.

---

## 5. Etapas de modelagem

> 🚧 *Pré-processamento e baseline concluídos (S3). Comparação de algoritmos na S4.*

**Seleção de features** — [`notebooks/02_selecao_features.ipynb`](notebooks/02_selecao_features.ipynb).
Das 35 colunas de `features_aluno`, **24 entram no modelo** e 11 ficam de fora, cada uma
com motivo e evidência no notebook:

| Grupo | Colunas | Motivo |
|---|---|---|
| Alvo | `alfabetizado` | Vira o `y` |
| Identificadores | `id_aluno`, `id_escola`, `id_municipio` | Rótulos sem significado; `id_municipio` define os grupos da divisão |
| Constantes | `ano`, `rede_codigo` | Um único valor em toda a base |
| Peso amostral | `peso_aluno` | Descreve o desenho da avaliação, não o aluno |
| Descartadas | `caderno`, `uf_anomala` | `caderno` só existe no dia da prova; `uf_anomala` é idêntica a `sigla_uf == "RS"` |
| Redundantes | `mun_meta_ano_alvo`, `mun_idade_mediana` | Correlação acima de 0,9 com outras features (a meta é calculável a partir da taxa e da distância) |

As decisões ficam em [`src/config/features_modelo.json`](src/config/features_modelo.json),
lido pelo notebook de modelagem: a lista de features existe em um único lugar.

**Alvo invertido.** `1 = não alfabetizado`. Precisão, recall e F1 passam a medir a classe
de risco, que é a que a política pública precisa encontrar.

**Divisão por município** — [`notebooks/03_modelagem_supervisionada.ipynb`](notebooks/03_modelagem_supervisionada.ipynb).
`GroupShuffleSplit` com 20% dos municípios no teste: treino com 4.360 municípios
(1.320.423 alunos) e teste com 1.090 (290.331), sem município em comum. Taxa de risco de
40,9% e 39,3%. O teste fica reservado para a avaliação final; a comparação de modelos usa
`GroupKFold` de 5 rodadas dentro do treino.

**Pipeline.** Pré-processamento e modelo num único `Pipeline` do scikit-learn, aprendendo
só com o treino de cada rodada:

| Grupo | Imputação | Transformação |
|---|---|---|
| 6 numéricas muito assimétricas (população, densidade, PIB per capita…) | Mediana | `log1p` + padronização |
| 15 numéricas | Mediana | Padronização |
| `regiao`, `sigla_uf` | Mais frequente | One-hot |
| `mun_elegivel_meta` | Constante 0 | — |

**Estratégia de amostragem.** A base completa é usada, sem reamostragem: o alvo é
relativamente balanceado (59,4% × 40,6%), e a classe de risco recebe mais peso com
`class_weight="balanced"`.

## 6. Escolha do algoritmo

> ⏳ *A preencher na S4.*

| Algoritmo | Por que foi considerado | Resultado |
|---|---|---|

Cobrir a decisão final com o trade-off explícito entre desempenho e interpretabilidade.
Num contexto de política pública, modelo que acerta mais e não se explica pode ser a
escolha errada.

---

## 7. Métricas de avaliação

> 🚧 *Baseline avaliado (S3). Curvas ROC e de precisão-recall, e o contraste entre
> divisão aleatória e por município, na S4 e S6.*

**Métrica prioritária: recall da classe de risco.** O falso negativo, deixar de sinalizar
uma criança em risco, é o erro mais grave. O recall é acompanhado de precisão e F1, porque
um modelo que apontasse risco para todos teria recall de 100% e nenhuma utilidade. A
acurácia não serve sozinha: o modelo que responde sempre "alfabetizado" acerta 59%.

**Baseline** (validação cruzada por município, 5 rodadas, treino):

| Modelo | ROC-AUC | Recall | Precisão | F1 | Acurácia |
|---|---|---|---|---|---|
| Ingênuo (classe mais comum) | 0,500 | 0,000 | 0,000 | 0,000 | 0,591 |
| Regressão logística | **0,661** | **0,652** | 0,516 | 0,573 | 0,607 |

Das 540.192 crianças em risco no treino, a regressão logística encontra 353.852 (65,5%).
Dos alertas que emite, 51,5% estão corretos, contra 41% se fossem dados ao acaso. Tabela
completa em [`reports/modelagem/`](reports/modelagem/) e matriz de confusão em
[`images/matriz_confusao_baseline.png`](images/matriz_confusao_baseline.png).

## 8. Interpretação dos resultados

> ⏳ *A preencher na S6.*

Feature Importance e SHAP. A leitura precisa ser causalmente honesta: o modelo mostra
associação, não efeito. Infraestrutura escolar aparecer como relevante não prova que
construir bibliotecas alfabetiza.

---

## 9. Insights encontrados

Dois achados da Fase 2 seguem como candidatos a desdobramento na modelagem:

- O gradiente de infraestrutura — quase 10 pontos percentuais de taxa média entre o
  primeiro e o quarto quartil do índice, no grão do município
- O fenômeno de reversão em 1.917 municípios, mais que o triplo dos que avançam devagar
  demais

A EDA da Fase 3 (`notebooks/01_eda.ipynb`), já no grão do aluno, confirma e refina:

- O gradiente de infraestrutura se mantém, mas mais fraco — 6,9 p.p. entre Q1 e Q4 (contra
  9,3 p.p. por município) — o efeito é real, mas parcialmente diluído aluno a aluno
- Municípios marcados como `uf_anomala` (RS) têm taxa de 45,3% contra 59,9% no resto do
  país — decidir incluir, marcar ou excluir essas linhas fica para a modelagem
- Nulos pequenos (até 2,6%) em 8 colunas de contexto municipal, a tratar por imputação

> ⏳ *Aprofundar na S6 e S7, com Feature Importance e SHAP.*

---

## 10. Limitações do projeto

**Sem features no grão da escola.** O `id_escola` de `fato_aluno` não é o código INEP: os
42.497 valores começam todos com `60`, prefixo que não corresponde a nenhuma UF. É
identificador sintético, atribuído pela avaliação para impedir reidentificação de
crianças em escolas pequenas — o join com `fato_escola` retorna **zero linhas**,
verificado.

A limitação é real e reduz o poder do modelo: variação entre escolas do mesmo município
costuma explicar bastante em educação. Mas a causa é legítima, e o projeto respeitou a
proteção de privacidade em vez de contorná-la. O contexto escolar entra agregado por
município, em `mun_indice_infraestrutura` e `mun_alunos_por_docente`.

**Cobertura temporal de dois anos.** A avaliação cobre 2023 e 2024. Dois pontos sustentam
comparação, não tendência.

**Defasagem das fontes econômicas.** O PIB total é de 2023 e a composição setorial de
2021, contra a avaliação de 2024. Riqueza municipal muda devagar, mas a defasagem existe.

**Anomalia do Rio Grande do Sul.** Documentada na Fase 2: 89,6% dos municípios gaúchos
com queda, mediana de −19,8, deslocamento uniforme que indica alteração na origem do
dado. As 61.053 linhas do estado são marcadas com `uf_anomala`, não removidas — a decisão
de excluí-las é de quem modela.

**Pseudo-replicação das variáveis municipais.** Tamanho efetivo de amostra de 5.450 para
features municipais, contra 1,6 milhão de linhas.

**Um município com composição setorial anulada.** Faina, em Goiás, com valor adicionado
industrial negativo de −477 milhões em 2021 — provável ajuste contábil do IBGE. As
proporções ficam nulas em vez de inverter de sinal.

**Features quase todas municipais.** Das 24 features, nenhuma descreve o aluno: crianças
do mesmo município recebem praticamente a mesma previsão. O modelo ranqueia municípios
mais do que alunos, o que limita o poder de separação (ROC-AUC de 0,66 no baseline).

**Percentuais acima de 100%.** `mun_pct_rural` e `mun_pct_transporte` passam de 100% em
1.279 (23%) e 1.105 (20%) municípios, com máximos de 552,6% e 467,0%. A frequência indica
erro no cálculo dessas colunas; foram mantidas no baseline, e a interpretação delas deve
ser feita com cautela.
---

## 11. Aplicação prática para políticas públicas

> ⏳ *A preencher na S7.*

**Ressalva de uso responsável**, a desenvolver: predição individual sobre crianças tem
risco de estigmatização. O uso legítimo é alocar apoio, não rotular aluno.

---

## 12. Possíveis evoluções futuras

> ⏳ *A preencher na S7.*

Candidatos já identificados:

- Séries temporais, quando houver mais edições da avaliação
- **MUNIC — Pesquisa de Informações Básicas Municipais** (IBGE): capacidade institucional
  da prefeitura. Avaliada e adiada por custo de exploração — 306 colunas e edições de
  anos distintos
- Features no grão da escola, se surgir chave que ligue a avaliação ao Censo Escolar
- Reaprendizado periódico e publicação do modelo como serviço

---

## 13. Como executar

### Pré-requisitos

- Python 3.11
- **Terraform ≥ 1.5** — a infraestrutura AWS é declarada em `infra/terraform/`
- Projeto no GCP com a API do BigQuery habilitada
- Conta AWS com acesso ao bucket do data lake da Fase 2

### Instalação

```bash
git clone https://github.com/amandacleite/tech-challenge-fase3.git
cd tech-challenge-fase3

python -m venv .venv
source .venv/Scripts/activate    # Linux e macOS: source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env             # preencher com os valores do seu ambiente
```

### Variáveis de ambiente

| Variável | Descrição |
|---|---|
| `GCP_PROJECT_ID` | Projeto GCP que fatura as consultas ao BigQuery |
| `AWS_REGION` | Região do bucket — `us-east-1` no AWS Academy |
| `AWS_BUCKET` | Bucket do data lake, compartilhado com a Fase 2 |
| `PIPELINE_ENV` | `dev` ou `prod` |

> ⚠️ **Nenhuma credencial vai para o Git.** O `.env` está no `.gitignore`; o
> `.env.example` traz apenas os nomes das variáveis.

### Configuração da AWS

No AWS Academy, as credenciais expiram a cada sessão do laboratório. Copie o bloco de
**AWS Details → AWS CLI → Show**:

```bash
aws configure
```

Em ambientes AWS Academy configure também:

```bash
aws configure set aws_session_token <TOKEN>
```

Validação:

```bash
aws sts get-caller-identity
aws s3 ls s3://<bucket>/
```

O segundo comando precisa listar `bronze/`, `silver/` e `gold/` — as camadas da Fase 2,
que esta fase consome. Se o bucket não existir, o laboratório foi reiniciado e a Fase 2
precisa ser reconstruída antes.

### Configuração do Google Cloud

Application Default Credentials, sem arquivo de chave permanente:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project SEU_PROJETO
```

Validação:

```bash
python src/ingestion/buscar_dataset.py censo
```

Deve listar os datasets do IBGE e do INEP disponíveis na Base dos Dados.

### Terraform

Instale o binário, caso não tenha:

```bash
winget install HashiCorp.Terraform   # Windows
brew install terraform               # macOS
```

Feche e reabra o terminal depois de instalar — o PATH só é lido na inicialização. Confirme
com `terraform version`.

**Não é necessário `terraform.tfvars`.** O ARN da role é montado em tempo de execução a
partir de `data.aws_caller_identity`, e os demais valores têm default em `variables.tf`.
Para usar outro bucket ou prefixo, passe `-var` na linha de comando — **o default é o
bucket do autor original, não o seu**; use sempre `-var="bucket=<seu-bucket>"` (e a mesma
variável `BUCKET=<seu-bucket>` ao rodar os scripts de `infra/` e `scripts/consultar.sh`).

Se o bucket for trocado **depois** que o crawler já existir, o Terraform não consegue
atualizar o `s3_target` in-place (o recrawl `CRAWL_NEW_FOLDERS_ONLY` o torna imutável) —
use `terraform apply -replace="aws_glue_crawler.bronze"` nesse caso.

> ⚠️ O `terraform.tfstate` e o `.terraform/` estão no `.gitignore` de
> `infra/terraform/`. O *state* descreve a infraestrutura e o ID da conta.

No AWS Academy não é possível criar roles IAM — a `LabRole` já vem provisionada.

### Execução

**1. Sincronizar as camadas da Fase 2** — baixa as tabelas que a Gold desta fase consome:

```bash
python src/ingestion/sincronizar_fase2.py --listar    # confere o que existe
python src/ingestion/sincronizar_fase2.py
```

Validação:

```bash
ls data/raw/
cat reports/ingestao/manifesto.jsonl
```

**2. Ingerir as fontes socioeconômicas** — dry run antes, sem custo:

```bash
python src/ingestion/ingestao_censo_2022.py --medir
python src/ingestion/ingestao_censo_2022.py

python src/ingestion/ingestao_pib.py --medir
python src/ingestion/ingestao_pib.py
```

Validação:

```bash
aws s3 ls s3://<bucket>/fase3/bronze/ --recursive --region us-east-1
```

**3. Criar a infraestrutura** — 6 recursos:

```bash
cd infra/terraform
terraform init
terraform apply
cd ../..
```

**4. Executar o fluxo** — crawler e Gold encadeados:

```bash
bash infra/executar_workflow.sh
```

O script dispara o Workflow, acompanha as duas etapas e mostra a saída no S3 ao final.

Para depurar uma etapa isolada, sem rodar o fluxo:

```bash
bash infra/executar_crawler.sh
aws glue start-job-run --job-name fase3_job_gold --region us-east-1
```

Validação:

```bash
aws s3 ls s3://<bucket>/fase3/gold/features_aluno/ --recursive --region us-east-1
bash scripts/consultar.sh verificacao_features_aluno
```

**5. Baixar a Gold para uso local nos notebooks** — evita depender de sessão AWS/Athena a
cada execução do EDA:

```bash
aws s3 cp s3://<bucket>/fase3/gold/features_aluno/ data/sample/features_aluno/ --recursive
```

**6. Construir o dataset de modelagem** ⏳

```bash
# python src/preprocessing/dataset.py
```

### Consultas analíticas

As consultas ficam versionadas em `sql/<camada>/`, com comentários explicando as
restrições que cada camada impõe. Quem clonar o repositório e tiver acesso ao Catalog
reproduz os mesmos números.

```bash
bash scripts/consultar.sh                             # lista as da Gold
bash scripts/consultar.sh verificacao_features_aluno  # executa uma
bash scripts/consultar.sh --todas                     # executa todas

CAMADA=bronze bash scripts/consultar.sh particoes_ingestao
```

### Explorar outras fontes

```bash
python src/ingestion/buscar_dataset.py <termo>             # localiza o dataset
python src/ingestion/explorar_fonte.py <dataset>           # lista tabelas e volumes
python src/ingestion/explorar_fonte.py <dataset> <tabela>  # detalha as colunas
```

### Reprocessar uma carga passada

O histórico é preservado por partição:

```bash
ls data/raw/censo_2022_municipio/          # lista as particoes disponiveis
cat reports/ingestao/manifesto.jsonl       # consulta, volume e data de cada execucao
```

A função `writer.ultima_particao(fonte, tabela)` resolve a carga mais recente. Para outra
data, aponte diretamente para a partição desejada.

### Notebooks

As dependências de notebook (`jupyterlab`, `ipykernel`) já estão no `requirements.txt`
principal — não é necessário instalar nada à parte.

---

## 14. Estrutura do repositório

```
.
├── data/
│   ├── raw/          # Bronze — <fonte>_<tabela>/dt_ingestao=... (NAO versionada)
│   ├── analytics/    # dataset de modelagem (NAO versionado)
│   └── sample/       # amostra estratificada (VERSIONADA)
├── images/           # figuras geradas, para o README
├── infra/
│   ├── terraform/    # databases, crawler, job, tabela e workflow
│   ├── executar_workflow.sh
│   └── executar_crawler.sh
├── notebooks/        # EDA, modelagem e interpretabilidade
├── reports/
│   ├── ingestao/     # manifesto de execucoes (VERSIONADO)
│   └── modelagem/    # metricas dos modelos (VERSIONADO)
├── scripts/
│   └── consultar.sh  # executor de consultas no Athena
├── sql/
│   ├── bronze/       # verificacao do historico de ingestao
│   └── gold/         # verificacao e exploracao do dataset
├── src/
│   ├── config/       # settings e features_modelo.json
│   ├── ingestion/    # busca, exploracao, extracao e sincronizacao
│   ├── preprocessing/# pipeline de transformacao para ML
│   ├── transformation/ # Glue Job da Gold
│   ├── modeling/     # treino, otimizacao, selecao
│   ├── evaluation/   # metricas, SHAP, comparacao
│   └── visualization/# graficos reutilizaveis
├── tests/
├── requirements.txt
└── README.md
```

**`data/sample/` é a exceção versionada**, de propósito: torna os notebooks executáveis
sem depender do S3, que expira junto com a sessão do laboratório.

---

## 15. Fluxo de trabalho Git

Nada é commitado direto na `main`. Toda branch entra por Pull Request.

### Branches

| Prefixo | Uso |
|---|---|
| `feature/` | Funcionalidade nova |
| `fix/` | Correção |
| `docs/` | Documentação |
| `chore/` | Configuração e manutenção |

### Padrão de commits

[Conventional Commits](https://www.conventionalcommits.org/pt-br/), no imperativo e em
minúsculas:

```
feat: ingere censo 2022 e pib municipal com particao por data
fix: separa ano do pib total do ano da composicao setorial
docs: documenta tratamento de vazamento de dados
```

### Pull Requests

A descrição explica **o que** foi feito, **por que** dessa forma, **como validar** e **o
que ficou de fora**. O PR é o registro da evolução do projeto — descrição vaga hoje é
contexto perdido amanhã.

---

## 16. Evidências de execução

| Evidência | Local | Status |
|---|---|---|
| Manifesto de ingestão | [`reports/ingestao/manifesto.jsonl`](reports/ingestao/manifesto.jsonl) | ✅ |
| Histórico de ingestão consultável por data | [`images/ConsultaAthenaCenso_2022_municipio.png`](images/ConsultaAthenaCenso_2022_municipio.png) | ✅ |
| Dados da Bronze no Athena | [`images/ConsultaAthenaCenso_2022_municipio_dados.png`](images/ConsultaAthenaCenso_2022_municipio_dados.png) | ✅ |
| Verificação de `features_aluno` no Athena | [`images/ConsultaAthenaFase3_gold.png`](images/ConsultaAthenaFase3_gold.png) | ✅ |
| Print — `terraform apply` | `images/` | ⏳ |
| Print — crawler com partição reconhecida | `images/` | ⏳ |
| Print — Glue Job da Gold concluído | `images/` | ⏳ |
| Notebook de EDA com saídas | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | ✅ |
| Notebook de modelagem | `notebooks/` | ⏳ |
| Matriz de confusão e curva ROC | `images/` | ⏳ |
| Gráficos SHAP | `images/` | ⏳ |
| Relatório técnico | `reports/` | ⏳ |
| Vídeo executivo (até 5 min) | — | ⏳ |
| Notebook de modelagem | [`notebooks/03_modelagem_supervisionada.ipynb`](notebooks/03_modelagem_supervisionada.ipynb) | 🚧 |
| Matriz de confusão e curva ROC | [`images/matriz_confusao_baseline.png`](images/matriz_confusao_baseline.png) | 🚧 |

A primeira delas é a que comprova a correção apontada na avaliação da Fase 2: a partição
por data de ingestão é reconhecida pelo Glue e consultável no Athena, então cargas
anteriores coexistem em vez de serem sobrescritas.

---

## 17. Roadmap e status

| Semana | Período | Entregável | Status |
|---|---|---|---|
| S0 | 11–13/set | Repositório, estrutura, dependências | ✅ |
| S1 | 14–20/set | Ingestão socioeconômica e catalogação | ✅ |
| S1 | 14–20/set | Gold no grão do aluno, com vazamento tratado | ✅ |
| S2 | 21–27/set | EDA | ✅ |
| S3 | 28/set–04/out | Pré-processamento e baseline | ✅ |
| S4 | 05–11/out | Modelagem — algoritmos comparados | ⏳ |
| S5 | 12–18/out | Otimização de hiperparâmetros | ⏳ |
| S6 | 19–25/out | Avaliação e interpretabilidade | ⏳ |
| S7 | 26–30/out | Clusterização, negócio e documentação | ⏳ |
| — | 31/out–03/nov | Margem para revisão | ⏳ |

---

## 18. Licença

Distribuído sob a licença MIT.

Os dados utilizados são públicos, produzidos pelo INEP e pelo IBGE e disponibilizados
pela plataforma Base dos Dados, sujeitos aos termos de uso originais de cada fonte.
