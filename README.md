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

**Em uma leitura.** Os dados públicos de alfabetização são anonimizados: todo aluno de uma
cidade recebe as mesmas variáveis. Por isso o modelo estima a chance de alfabetização de uma
criança **naquele contexto**, não de uma criança específica — e nessa pergunta ele acerta. A
probabilidade prevista fica a cerca de um ponto da observada, separa contextos de 13% a 65% de
risco e aponta os municípios que ficarão abaixo da meta com 90% de precisão entre os dez
primeiros. O que nenhum modelo consegue com esses dados é distinguir duas crianças da mesma
cidade: 88% da variação está aí.

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

**O aluno provável.** O enunciado pede prever a alfabetização "utilizando variáveis
educacionais, territoriais e socioeconômicas" — e todas elas descrevem o município, não a
criança. Alunos da mesma cidade recebem o mesmo vetor de features (seção 7). Por isso o
modelo estima a **probabilidade de alfabetização para um aluno inserido em determinado
contexto**: a resposta para "um aluno deste território, desta condição socioeconômica e
desta condição educacional se alfabetiza?". Para essa pergunta, a métrica central é a
calibração — quando o modelo diz 40% de risco, cerca de 40% dos alunos daquele contexto de
fato não se alfabetizam —, reportada ao lado das métricas de classificação.

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

> ✅ *Pré-processamento, baseline e comparação de algoritmos concluídos. Calibração na seção 7.*

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
| 11 numéricas | Mediana | Padronização |
| 4 razões do Censo Escolar com valores impossíveis | Mediana | Teto no percentil 99 do treino + padronização |
| `regiao`, `sigla_uf` | Mais frequente | One-hot |
| `mun_elegivel_meta` | Constante 0 | — |

**Estratégia de amostragem.** A base completa é usada, sem reamostragem: o alvo é
relativamente balanceado (59,4% × 40,6%), e a classe de risco recebe mais peso com
`class_weight="balanced"`. O peso balanceado tem um efeito colateral sobre as
probabilidades, tratado na seção 7.

**Teto nas razões do Censo Escolar.** `mun_alunos_por_turma`, `mun_alunos_por_docente`,
`mun_pct_rural` e `mun_pct_transporte` têm numerador e denominador de populações diferentes e
assumem valores impossíveis (seção 10). O teto é aprendido no `fit`, portanto só com o
treino, e configurado em `limitar_extremos` no JSON de decisões. O efeito nas métricas foi
nulo — ROC-AUC de 0,6709 com e sem teto na regressão logística, o modelo mais sensível a
extremos por ser linear. A correção é de consistência, para que o modelo não receba valores
impossíveis, não de desempenho.

## 6. Escolha do algoritmo

Quatro algoritmos, avaliados no mesmo conjunto de teste — 1.090 municípios nunca vistos no
treino —, cada um com o limiar escolhido na validação cruzada para o mesmo recall-alvo:

| Algoritmo | ROC-AUC | PR-AUC | Recall | Precisão | F1 | Brier calibrado |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0,674 | 0,552 | 0,665 | 0,509 | 0,577 | 0,2163 |
| Random Forest | 0,673 | 0,549 | 0,694 | 0,505 | 0,585 | 0,2168 |
| Regressão logística | 0,672 | 0,550 | 0,687 | 0,505 | 0,582 | 0,2166 |
| Árvore de decisão rasa | 0,664 | 0,528 | 0,605 | 0,517 | 0,557 | 0,2180 |

O Brier calibrado é medido sem o RS, após a correção de prior (seção 7).

**Os quatro empatam.** A diferença de ROC-AUC entre o melhor e o pior é de 0,010, e entre os
três primeiros, de 0,002 — dentro do que uma nova divisão treino/teste poderia inverter. Os
modelos de árvores não extraem mais sinal que a regressão logística, e a seção 7 mostra por
quê: o limite está nos dados, não no algoritmo.

**Consequência para a escolha.** Com desempenho equivalente, o critério passa a ser
interpretabilidade. Num contexto de política pública, modelo que acerta o mesmo e se explica
melhor é a escolha certa.

> 🚧 *Decisão final do grupo pendente.*

---

## 7. Métricas de avaliação

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

### O limite está nos dados

Das 29 features de `features_aluno`, **apenas uma varia entre alunos do mesmo município** — o
`caderno`, descartado por só existir no dia da prova. Todo aluno de uma cidade recebe o mesmo
vetor de features, e nenhum modelo pode distinguir duas crianças da mesma cidade.

No teste — 280.021 alunos em 1.009 municípios, sem o RS —, a variação do alvo se divide assim:

| Onde está a variação | Fração |
|---|---:|
| Entre municípios — alcançável pelas features | 11,6% |
| Dentro de cada município — inalcançável | 88,4% |

Para medir o que esse limite permite, imaginamos o melhor previsor possível com informação
municipal: um que soubesse a taxa real de 2024 de cada município. Ele não existe, mas define
o teto de qualquer modelo construído com estes dados.

### Duas medidas, dois tetos

O modelo responde a duas perguntas diferentes, e cada uma tem sua métrica.

| O que se mede | Métrica | Modelo | Teto | Alcançado |
|---|---|---:|---:|---:|
| Ordenar alunos por risco | ROC-AUC | 0,674 | 0,699 | 88% |
| Acertar a probabilidade, antes da calibração | Brier skill | 6,5% | 11,6% | 56% |
| Acertar a probabilidade, depois da calibração | Brier skill | 8,9% | 11,6% | **77%** |

"Alcançado" é a fração do ganho possível que o modelo obtém: sobre o acaso, no ROC-AUC; sobre
prever sempre a média, no Brier.

**Na ordenação, o modelo já estava perto do teto.** Uma única feature, a taxa do ano anterior,
dá 0,663; o modelo completo, 0,674; o melhor possível, 0,699. Outro algoritmo ou mais busca de
hiperparâmetros disputam no máximo 0,025 — e a seção 6 mostra que os quatro modelos empatam.

**Nas probabilidades, estava longe — e a causa era corrigível.** É a calibração, abaixo.

O teto do Brier, 11,6%, coincide com a variação entre municípios, 11,6%. Não por acaso: o
melhor previsor municipal elimina exatamente a parte da variação que está entre os municípios,
e nada da que está dentro deles. Duas métricas independentes chegam ao mesmo limite.

### Calibração

Implementada em [`src/evaluation/calibracao.py`](src/evaluation/calibracao.py), sobre as
probabilidades já gravadas, sem retreinar nenhum modelo.

**Os quatro modelos superestimavam o risco em 6 a 8 pontos, em todas as faixas.** A causa é o
`class_weight="balanced"`, escolha deliberada para priorizar o recall: o modelo aprende como se
metade dos alunos fosse de risco, quando a taxa real no treino é 40,9%.

A correção é analítica: reescalar as chances pela taxa de risco do **treino** — nada do teste
entra. Por ser monotônica, ninguém muda de posição.

| Modelo | ECE antes | ECE depois | Brier antes | Brier depois |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 7,53 | 0,90 | 0,2220 | 0,2163 |
| Regressão logística | 7,79 | 0,76 | 0,2227 | 0,2166 |
| Random Forest | 8,07 | 1,26 | 0,2232 | 0,2168 |
| Árvore de decisão rasa | 8,22 | 0,84 | 0,2249 | 0,2180 |

ECE em pontos percentuais: a distância média entre o risco previsto e o observado.

Depois da correção, a previsão fica a cerca de um ponto do observado em toda a faixa:

| Decil de risco previsto | Previsto | Observado |
|---|---:|---:|
| 1 — contextos mais favoráveis | 14,7% | 13,3% |
| 5 | 36,6% | 38,2% |
| 10 — contextos mais adversos | 62,8% | 64,9% |

**86,7% das crianças se alfabetizam nos contextos mais favoráveis; 35,1%, nos mais adversos.**
Diagrama em [`images/calibracao_teste.png`](images/calibracao_teste.png).

### O que muda, e o que não muda

**Não muda a classificação.** ROC-AUC, recall, precisão e F1 são idênticos antes e depois: a
correção é monotônica, e o limiar convertido produz as mesmas decisões. Quem olhar só a tabela
da seção 6 não vê diferença.

**Muda a confiança na probabilidade — e ela importa para uma pergunta de negócio, não para a
outra.** Para saber *quem fica abaixo da meta*, a taxa prevista precisa estar certa em valor: com
as probabilidades originais, 715 dos 889 municípios do teste seriam sinalizados, quando 373 de
fato ficaram abaixo; com as calibradas, 322. Para saber *por onde começar*, basta a ordem — e ela
não depende da calibração: a precisão entre os dez municípios de maior risco é de 90% nas duas
versões (seção 11).

**Não muda o limite.** Nenhum ajuste distingue duas crianças da mesma cidade. Romper o teto
exigiria variáveis individuais, e a seção 10 documenta por que nenhuma fonte pública as oferece.

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

### Regiões com padrões semelhantes

Resposta à pergunta de negócio do enunciado, em
[`src/modeling/clusterizacao.py`](src/modeling/clusterizacao.py). K-Means sobre o **contexto**
dos municípios — socioeconômico, educacional e territorial —, com a taxa de alfabetização
usada **só depois**, para descrever cada grupo. Agrupar pelo resultado tornaria tautológica
a conclusão de que contextos parecidos têm resultados parecidos.

O número de grupos foi escolhido pela silhueta, máxima em k = 4 — mas baixa, 0,188, e a
curva de inércia não tem cotovelo. Os contextos municipais formam um contínuo, não grupos
nítidos: os quatro grupos servem como tipologia, não como separação natural.

| Grupo | Municípios | Alunos | Alfabetização | Perfil |
|---|---:|---:|---:|---|
| 1 | 335 | 96.769 | 49,2% | Rural isolado, menor infraestrutura e densidade |
| 2 | 1.783 | 269.329 | 59,2% | Menor PIB e menor alfabetização adulta; mais tempo integral |
| 3 | 1.252 | 1.091.654 | 59,5% | Urbano, maior PIB, alfabetização adulta e infraestrutura |
| 4 | 2.080 | 153.002 | 71,9% | Interior estruturado, turmas menores |

Taxa ponderada pelos alunos, sem o RS.

**O grupo urbano mais rico alfabetiza menos que o interior estruturado.** O grupo 3 tem o
melhor contexto em quase tudo — PIB per capita mediano de R$ 48 mil, 95,4% de alfabetização
adulta — e chega a 59,5%. O grupo 4, com contexto um pouco inferior, alcança 71,9%. São doze
pontos contra o contexto, no grupo que concentra 1,09 milhão dos 1,6 milhão de alunos. A
capital paulista, maior município do grupo 3, alfabetiza 48,5%.

**O grupo do Nordeste empata com as grandes cidades com o pior contexto.** O grupo 2 tem o
menor PIB per capita mediano, R$ 14,9 mil, e a menor alfabetização adulta, 79,4% — e chega
aos mesmos 59,2% do grupo urbano. É também o grupo com mais tempo integral, 20,6% das
matrículas.

| Região | Grupo 1 | Grupo 2 | Grupo 3 | Grupo 4 |
|---|---:|---:|---:|---:|
| Centro-Oeste | 11% | 4% | 22% | 63% |
| Nordeste | 4% | 84% | 9% | 2% |
| Norte | 43% | 25% | 9% | 23% |
| Sudeste | 0% | 9% | 39% | 51% |
| Sul | 1% | 0% | 28% | 71% |

Sul, Sudeste e Centro-Oeste se distribuem pelos mesmos grupos e têm padrões semelhantes. O
Nordeste é o mais homogêneo, com 84% dos municípios num grupo só. O Norte é o mais
heterogêneo, espalhado por três grupos.

**Leitura causal, não.** O grupo 4 tem as menores turmas — 18,2 alunos, contra 22,2 no grupo
3 —, e é tentador atribuir a diferença a isso. O agrupamento mostra associação; a explicação
exigiria outro desenho de estudo. Projeção em
[`images/clusters_pca.png`](images/clusters_pca.png) e escolha de k em
[`images/clusters_silhueta.png`](images/clusters_silhueta.png).

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

**Fontes alternativas foram buscadas e descartadas.** O Indicador de Nível Socioeconômico
(INSE) do INEP, construído a partir do questionário do Saeb, seria a candidata ideal, mas
está no grão da escola e não pode ser ligado aos alunos avaliados. Também foi testado se o
código sintético da escola se mantinha entre 2023 e 2024, o que permitiria usar o resultado
da escola no ano anterior: das 36.462 escolas com o mesmo código nos dois anos, **35.597
estavam em municípios diferentes** — o código é regenerado a cada ano, e a coincidência é
de numeração, não de escola. A anonimização decorre da política do INEP de proteção de
dados, em observância à LGPD.

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
do mesmo município recebem a mesma previsão. Como 88% da variação do alvo está dentro dos
municípios, o ROC-AUC tem teto de 0,699 com qualquer conjunto de features municipais
(seção 7). O modelo responde bem à probabilidade para o aluno de um contexto; não distingue
crianças da mesma cidade.

**Razões do Censo Escolar com valores impossíveis.** Quatro variáveis herdadas da Fase 2
têm numerador e denominador de populações diferentes:

| Variável | Causa | Extensão |
|---|---|---|
| `mun_pct_rural`, `mun_pct_transporte` | Numerador conta matrículas de todas as etapas; denominador, só anos iniciais | Acima de 100% em 1.279 e 1.105 municípios; máximos de 552,6% e 467,0% |
| `mun_alunos_por_turma`, `mun_alunos_por_docente` | Turmas multisseriadas de áreas indígenas não contam como turma de anos iniciais | Jordão (AC), com 44,7% de população indígena: 109,9 alunos por turma, contra mediana de 20,3 |

As variáveis continuam crescendo com o fenômeno que medem e funcionam como índice, não como
percentual. O pipeline aplica um teto no percentil 99 do treino (seção 5), e o efeito nas
métricas foi nulo — a amostragem de até 30 alunos por município já limitava o peso de casos
como Jordão. A correção na origem fica pendente.

---

## 11. Aplicação prática para políticas públicas

### Municípios que podem não atingir a meta

Resposta à pergunta de negócio do enunciado, em
[`src/evaluation/metas_municipios.py`](src/evaluation/metas_municipios.py). Como as features
são do município, a média das probabilidades calibradas dos alunos de uma cidade estima a
taxa de alfabetização daquela cidade — e, comparada com a meta pactuada, indica quem tende a
ficar abaixo.

**O que a calibração muda aqui.** O backtest roda com as duas versões das probabilidades:

| Medida | Original | Calibrada |
|---|---:|---:|
| Municípios sinalizados abaixo da meta — reais: 373 | 715 | 322 |
| Erro médio da taxa prevista | 8,9 pontos | 5,7 pontos |
| Recall | 94,9% | 57,1% |
| Precisão nos 10 de maior risco | 90% | 90% |

A versão original sinaliza oito em cada dez municípios: quase não deixa escapar quem fica
abaixo, porque aponta quase todos — e por isso tem recall e F1 maiores. A calibrada sinaliza um
número próximo do real e erra a taxa em três pontos a menos. A ordem de prioridade é
praticamente a mesma nas duas: **a calibração corrige o valor, não a ordem.**

**Backtest.** O modelo usa contexto de 2023 para prever 2024, então a previsão pode ser
conferida contra o que de fato aconteceu. Em 889 municípios do teste — nunca vistos no
treino, sem RS, com meta publicada e ao menos 30 alunos avaliados:

| Medida | Valor |
|---|---:|
| Correlação entre taxa prevista e observada | 0,815 |
| Erro médio da taxa | 5,7 pontos |
| Municípios de fato abaixo da meta | 373 (42%) |
| Precisão nos 10 de maior risco previsto | 90% |
| Precisão nos 50 de maior risco | 86% |
| Precisão nos 100 de maior risco | 83% |

A acurácia geral, de 69,7%, subestima o que importa. Perto da meta, poucos pontos invertem
a classificação, e é ali que se concentram os erros. Um gestor não usa a lista inteira:
prioriza o topo, onde o acerto dobra a taxa-base de 42%. Gráfico em
[`images/metas_backtest_municipios.png`](images/metas_backtest_municipios.png).

**Para metas futuras**, o mesmo procedimento se aplica com o contexto de 2024 para projetar
2025. O backtest é o que dá confiança para usá-lo adiante.

### Uso responsável

O modelo estima a probabilidade para o aluno de um contexto, não o destino de uma criança.
Usá-lo para rotular alunos seria um erro de interpretação, além de um risco de
estigmatização. O uso legítimo é **alocar apoio a territórios e redes**, onde a estimativa é
calibrada e o ranking de municípios tem precisão demonstrada.

---

## 12. Possíveis evoluções futuras

> 🚧 *Em construção.*

Candidatos já identificados:

- Séries temporais, quando houver mais edições da avaliação
- **MUNIC — Pesquisa de Informações Básicas Municipais** (IBGE): capacidade institucional
  da prefeitura. Avaliada e adiada por custo de exploração — 306 colunas e edições de
  anos distintos
- **INSE e indicadores no grão da escola**, se a anonimização permitir ligá-los aos alunos
  avaliados — é o único caminho identificado para romper o teto de 0,699 de ROC-AUC
- **Correção na origem das razões do Censo Escolar**, com numerador e denominador da mesma
  etapa
- **Recalibração por regressão isotônica**, ajustada em previsões fora da amostra do treino,
  se a correção analítica de prior deixar de bastar em outros recortes
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

**6. Treinar e avaliar os modelos** — notebooks `03` a `06`, que gravam métricas e
probabilidades de teste em `reports/modelagem/`.

**7. Calibração, metas e agrupamentos** — sobre as probabilidades gravadas, sem retreinar:

```bash
python -m src.evaluation.calibracao        # calibração e correção de prior
python -m src.evaluation.metas_municipios  # backtest das metas por município
python -m src.modeling.clusterizacao       # grupos de municípios por contexto
```

A ordem importa: `metas_municipios` lê as probabilidades calibradas que `calibracao` grava.

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
│   ├── modeling/     # treino, otimizacao, selecao e clusterizacao
│   ├── evaluation/   # metricas, SHAP, comparacao, calibracao e metas
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
| Notebooks de modelagem | [`notebooks/03`](notebooks/03_modelagem_supervisionada.ipynb) a [`06`](notebooks/06_arvore_decisao.ipynb) | ✅ |
| Matriz de confusão e curva ROC | [`images/hgb_avaliacao_teste.png`](images/hgb_avaliacao_teste.png) | ✅ |
| Gráficos SHAP | [`images/hgb_shap_summary.png`](images/hgb_shap_summary.png) | ✅ |
| Calibração antes e depois da correção | [`images/calibracao_teste.png`](images/calibracao_teste.png) | ✅ |
| Backtest das metas por município | [`images/metas_backtest_municipios.png`](images/metas_backtest_municipios.png) | ✅ |
| Agrupamento de municípios | [`images/clusters_pca.png`](images/clusters_pca.png) | ✅ |
| Relatório técnico | `reports/` | ⏳ |
| Vídeo executivo (até 5 min) | — | ⏳ |

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
| S4 | 05–11/out | Modelagem — algoritmos comparados | ✅ |
| S5 | 12–18/out | Otimização de hiperparâmetros | ✅ |
| S6 | 19–25/out | Avaliação, calibração e interpretabilidade | 🚧 |
| S7 | 26–30/out | Clusterização e perguntas de negócio | ✅ |
| S7 | 26–30/out | Documentação e vídeo executivo | 🚧 |
| — | 31/out–03/nov | Margem para revisão | ⏳ |

---

## 18. Licença

Distribuído sob a licença MIT.

Os dados utilizados são públicos, produzidos pelo INEP e pelo IBGE e disponibilizados
pela plataforma Base dos Dados, sujeitos aos termos de uso originais de cada fonte.
