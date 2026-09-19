"""
Camada Gold da Fase 3 — Glue Job (PySpark).

Constrói `features_aluno`: uma linha por estudante avaliado em 2024, com
contexto de escola e de município, pronta para modelagem supervisionada.

A Gold existe para entregar dado pronto para consumo. Como o consumo desta
fase é um modelo no grão do aluno, a junção hierárquica acontece aqui — não
num script de pré-processamento. Feita uma vez, validada, e qualquer
consumidor futuro recebe a tabela pronta.

Fontes, todas lidas pelo Glue Catalog:

    alfabetizacao_silver.fato_aluno            alvo e atributos do estudante
    alfabetizacao_silver.dim_territorio        UF e regiao
    alfabetizacao_gold.features_municipio      contexto educacional
    fase3_bronze.censo_2022_municipio          demografia e domicilios
    fase3_bronze.pib_municipal_municipio       economia

Quatro decisões atravessam o módulo, e cada uma tem consequência:

**Apenas 2024.** O contexto municipal vem de 2023 — informação disponível
no momento em que a criança de 2024 seria avaliada. Usar o agregado do
mesmo ano vazaria o resultado da própria turma para dentro da predição
individual. O corte reduz o volume à metade e torna o modelo genuinamente
preditivo.

**As colunas de vazamento não existem na saída.** `proficiencia`,
`distancia_corte` e `faixa_proximidade` definem o alvo por construção —
`alfabetizado` é `proficiencia >= 743`, verificado em 3.354.661 registros.
Excluí-las aqui, e não no consumo, impede o uso acidental.

**`aluno_valido` é filtrado, não marcado.** Diferente da Silver: aluno sem
prova não tem alvo, e a Gold entrega dado pronto. Já o Rio Grande do Sul é
marcado e não removido — a anomalia é de uma fonte específica, e um
consumidor de dashboard pode querer o estado.

**`id_municipio` sai do conjunto de features e fica como chave de
agrupamento.** Com 5.570 categorias, qualquer codificação memorizaria o
município. Ele é indispensável para a divisão treino/teste por grupo, que é
o que evita o vazamento de contexto.

**Sem features no grão da escola.** O `id_escola` de `fato_aluno` não é o
código INEP: os 42.497 valores começam todos com `60`, prefixo que não
corresponde a nenhuma UF. É identificador sintético, atribuído pela
avaliação para impedir reidentificação de crianças em escolas pequenas — o
join com `fato_escola` retorna zero linhas, verificado.

O contexto escolar entra agregado por município, em `mun_indice_infraestrutura`
e `mun_alunos_por_docente`. Perde-se a variação entre escolas do mesmo
município, e a limitação está registrada no README.

Parâmetros do Job:
    --JOB_NAME          nome do job (injetado pelo Glue)
    --BUCKET_DESTINO    bucket de escrita
    --DATABASE_SILVER   database da Silver da Fase 2
    --DATABASE_GOLD     database da Gold da Fase 2
    --DATABASE_FASE3    database da Bronze da Fase 3
    --ENV               dev | prod
"""

import sys

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# ===========================================================================
# CONTRATO
# ===========================================================================

# Ano da avaliação que o modelo prevê
ANO_ALVO = 2024

# Ano do contexto municipal. Defasagem deliberada: é a informação que um
# gestor teria em mãos antes da avaliação do ano seguinte.
ANO_CONTEXTO = 2023

# A avaliação é do 2º ano do fundamental, e as metas municipais são da rede
# Municipal — o recorte casa com a Gold da Fase 2.
REDE_MUNICIPAL = "3"

# UF com anomalia documentada na Fase 2: 89,6% dos municípios com queda,
# mediana de -19,8, deslocamento uniforme que indica alteração na origem do
# dado. Marcada, não removida.
UF_ANOMALA = "RS"

# Definem o alvo por construção. Não entram na saída.
COLUNAS_VAZAMENTO = [
    "proficiencia",
    "distancia_corte",
    "faixa_proximidade",
]

TABELAS = {
    "aluno": ("silver", "fato_aluno"),
    "territorio": ("silver", "dim_territorio"),
    "municipio_educacional": ("gold", "features_municipio"),
    "censo": ("fase3", "censo_2022_municipio"),
    "pib": ("fase3", "pib_municipal_municipio"),
}

# ---------------------------------------------------------------------------
# Schema explícito da saída
# ---------------------------------------------------------------------------

ESQUEMA_GOLD = [
    # --- chaves e agrupamento
    ("ano", "int"),
    ("id_aluno", "string"),
    ("id_escola", "string"),
    # Chave de agrupamento para GroupKFold, não feature
    ("id_municipio", "string"),
    ("sigla_uf", "string"),
    ("regiao", "string"),
    # --- alvo
    ("alfabetizado", "boolean"),
    ("peso_aluno", "double"),
    # --- marcações
    ("uf_anomala", "boolean"),
    # --- atributos do aluno
    ("rede_codigo", "string"),
    ("caderno", "string"),
    # --- contexto educacional do município, ano anterior
    ("mun_taxa_ano_anterior", "double"),
    ("mun_total_escolas", "int"),
    ("mun_alunos_por_docente", "double"),
    ("mun_alunos_por_turma", "double"),
    ("mun_pct_integral", "double"),
    ("mun_indice_infraestrutura", "double"),
    ("mun_pct_rural", "double"),
    ("mun_pct_transporte", "double"),
    # --- contexto socioeconômico
    ("mun_populacao", "int"),
    ("mun_densidade", "double"),
    ("mun_moradores_por_domicilio", "double"),
    ("mun_taxa_alfabetizacao_adulta", "double"),
    ("mun_idade_mediana", "double"),
    ("mun_indice_envelhecimento", "double"),
    ("mun_pct_indigena", "double"),
    ("mun_pct_quilombola", "double"),
    ("mun_pib_per_capita", "double"),
    ("mun_pct_va_agropecuaria", "double"),
    ("mun_pct_va_industria", "double"),
    ("mun_pct_va_servicos", "double"),
    ("mun_pct_va_administracao_publica", "double"),
]

DESTINO = "gold/features_aluno"


# ===========================================================================
# Construção
# ===========================================================================


def preparar_aluno(df: DataFrame) -> DataFrame:
    """
    Base do dataset: estudantes válidos do ano-alvo, na rede Municipal.

    `aluno_valido` é filtro e não marcação: quem não fez a prova não tem
    alvo, e a Gold entrega dado pronto para consumo.
    """

    return (
        df.filter(
            (F.col("ano") == ANO_ALVO)
            & F.col("aluno_valido")
            & (F.col("rede_codigo") == REDE_MUNICIPAL)
            & F.col("alfabetizado").isNotNull()
        )
        .select(
            "ano",
            "id_aluno",
            "id_escola",
            "id_municipio",
            "rede_codigo",
            "caderno",
            "alfabetizado",
            "peso_aluno",
        )
    )


def preparar_municipio_educacional(df: DataFrame) -> DataFrame:
    """
    Contexto educacional do município, com a taxa do ano anterior.

    `taxa_2023` é informação disponível antes da avaliação de 2024. A taxa
    do próprio ano é a média do alvo dos alunos daquele município e vazaria
    o resultado da turma para dentro da predição individual — por isso
    `alvo_taxa_2024` e `alvo_atingiu_meta` não aparecem aqui.
    """

    return df.select(
        "id_municipio",
        F.col("taxa_2023").alias("mun_taxa_ano_anterior"),
        F.col("total_escolas").alias("mun_total_escolas"),
        F.col("alunos_por_docente").alias("mun_alunos_por_docente"),
        F.col("alunos_por_turma").alias("mun_alunos_por_turma"),
        F.col("pct_matricula_integral").alias("mun_pct_integral"),
        F.col("indice_infraestrutura").alias("mun_indice_infraestrutura"),
        F.col("pct_matricula_rural").alias("mun_pct_rural"),
        F.col("pct_matricula_transporte").alias("mun_pct_transporte"),
    )


def preparar_socioeconomico(censo: DataFrame, pib: DataFrame) -> DataFrame:
    """
    Contexto demográfico e econômico do município.

    O Censo é de 2022 e o PIB de 2023, ambos anteriores à avaliação. São
    variáveis estruturais, que mudam devagar — a defasagem está registrada
    como limitação no README.
    """

    demografia = censo.select(
        "id_municipio",
        F.col("populacao").alias("mun_populacao"),
        (F.col("populacao") / F.col("area")).alias("mun_densidade"),
        (F.col("populacao") / F.col("domicilios")).alias(
            "mun_moradores_por_domicilio"
        ),
        F.col("taxa_alfabetizacao_adulta_censo").alias(
            "mun_taxa_alfabetizacao_adulta"
        ),
        F.col("idade_mediana").alias("mun_idade_mediana"),
        F.col("indice_envelhecimento").alias("mun_indice_envelhecimento"),
        (100 * F.col("populacao_indigena") / F.col("populacao")).alias(
            "mun_pct_indigena"
        ),
        (100 * F.col("populacao_quilombola") / F.col("populacao")).alias(
            "mun_pct_quilombola"
        ),
    )

    colunas_va = [
        "va_agropecuaria",
        "va_industria",
        "va_servicos",
        "va_adespss",
    ]

    # --- PIB total: o ano mais recente disponível
    #
    # O IBGE publica o PIB total antes da desagregação por atividade
    # econômica. Tomar o mesmo ano para os dois blocos deixa as quatro
    # proporções setoriais 100% nulas — erro que já ocorreu neste projeto.
    # Cada bloco escolhe o ano mais recente em que o seu dado existe.
    ano_pib = (
        pib.filter(F.col("pib").isNotNull())
        .agg(F.max("ano"))
        .collect()[0][0]
    )

    total = pib.filter(F.col("ano") == ano_pib).select("id_municipio", "pib")

    economia = total.join(
        demografia.select("id_municipio", "mun_populacao"),
        "id_municipio",
        "left",
    ).withColumn(
        # PIB absoluto faria o modelo aprender tamanho de município
        "mun_pib_per_capita",
        F.col("pib") / F.col("mun_populacao"),
    )

    # --- Composição setorial: o ano mais recente com valor adicionado
    com_va = pib
    for coluna in colunas_va:
        com_va = com_va.filter(F.col(coluna).isNotNull())

    ano_va = com_va.agg(F.max("ano")).collect()[0][0]

    setorial = com_va.filter(F.col("ano") == ano_va)

    soma_va = sum(F.col(c) for c in colunas_va)

    # O IBGE admite valor adicionado negativo — subsídios ou ajustes
    # maiores que a produção do setor. A soma pode ficar menor que as
    # partes e as proporções invertem de sinal, o que não é interpretável.
    valida = soma_va > 0
    for coluna in colunas_va:
        valida = valida & (F.col(coluna) >= 0)

    for coluna, nome in (
        ("va_agropecuaria", "mun_pct_va_agropecuaria"),
        ("va_industria", "mun_pct_va_industria"),
        ("va_servicos", "mun_pct_va_servicos"),
        ("va_adespss", "mun_pct_va_administracao_publica"),
    ):
        setorial = setorial.withColumn(
            nome, F.when(valida, 100 * F.col(coluna) / soma_va)
        )

    economia = economia.join(
        setorial.select(
            "id_municipio",
            "mun_pct_va_agropecuaria",
            "mun_pct_va_industria",
            "mun_pct_va_servicos",
            "mun_pct_va_administracao_publica",
        ),
        "id_municipio",
        "left",
    ).select(
        "id_municipio",
        "mun_pib_per_capita",
        "mun_pct_va_agropecuaria",
        "mun_pct_va_industria",
        "mun_pct_va_servicos",
        "mun_pct_va_administracao_publica",
    )

    return demografia.join(economia, "id_municipio", "left")


def aplicar_esquema(df: DataFrame) -> DataFrame:
    """
    Projeta no schema declarado.

    Colunas fora do contrato são descartadas — inclusive as de vazamento,
    que assim não podem ser usadas por acidente.
    """

    projecao = []

    for coluna, tipo in ESQUEMA_GOLD:
        if coluna in df.columns:
            projecao.append(F.col(coluna).cast(tipo).alias(coluna))
        else:
            projecao.append(F.lit(None).cast(tipo).alias(coluna))

    return df.select(*projecao)


def construir(fontes: dict) -> DataFrame:
    """Junta os três níveis e projeta no schema da Gold."""

    aluno = preparar_aluno(fontes["aluno"])
    educacional = preparar_municipio_educacional(fontes["municipio_educacional"])
    socioeconomico = preparar_socioeconomico(fontes["censo"], fontes["pib"])

    territorio = fontes["territorio"].select(
        "id_municipio", "sigla_uf", "regiao"
    )

    resultado = (
        aluno.join(territorio, "id_municipio", "left")
        .join(educacional, "id_municipio", "left")
        .join(socioeconomico, "id_municipio", "left")
    )

    resultado = resultado.withColumn(
        "uf_anomala", F.col("sigla_uf") == UF_ANOMALA
    )

    return aplicar_esquema(resultado)


def validar(df: DataFrame, logger) -> None:
    """
    Confere a saída antes de gravar, e interrompe quando o defeito é grave.

    A validação existe porque os dois problemas mais sérios encontrados
    nesta fase não geraram exceção nenhuma: quatro colunas 100% nulas por
    defasagem de publicação do IBGE, e um município com proporções
    setoriais invertidas por valor adicionado negativo. Ambos entrariam no
    modelo como dado legítimo.

    Regras bloqueantes derrubam o Job; alertas apenas registram.
    """

    logger("-" * 60)
    logger("Validacao da saida")

    total = df.count()
    bloqueios = []

    # --- bloqueante: chave do grão
    duplicados = total - df.select("ano", "id_aluno").distinct().count()

    logger(f"  chave (ano, id_aluno) duplicada:   {duplicados}")

    if duplicados:
        bloqueios.append(f"{duplicados} chaves duplicadas")

    # --- bloqueante: identificador territorial
    fora_padrao = df.filter(F.length("id_municipio") != 7).count()

    logger(f"  id_municipio fora de 7 digitos:    {fora_padrao}")

    if fora_padrao:
        bloqueios.append(f"{fora_padrao} id_municipio malformado")

    # --- bloqueante: alvo sem valor não tem o que prever
    alvo_nulo = df.filter(F.col("alfabetizado").isNull()).count()

    logger(f"  alvo nulo:                         {alvo_nulo}")

    if alvo_nulo:
        bloqueios.append(f"{alvo_nulo} linhas com alvo nulo")

    # --- bloqueante: coluna de vazamento na saída
    presentes = [c for c in COLUNAS_VAZAMENTO if c in df.columns]

    logger(f"  colunas de vazamento presentes:    {len(presentes)}")

    if presentes:
        bloqueios.append(f"vazamento: {', '.join(presentes)}")

    # --- alerta: percentual fora da faixa indica erro de cálculo
    percentuais = [c for c, _ in ESQUEMA_GOLD if c.startswith("mun_pct_")]

    fora_faixa = {}

    for coluna in percentuais:
        quantidade = df.filter(
            (F.col(coluna) < 0) | (F.col(coluna) > 100)
        ).count()

        if quantidade:
            fora_faixa[coluna] = quantidade

    if fora_faixa:
        logger(f"  ALERTA percentuais fora de 0-100:  {fora_faixa}")
    else:
        logger(f"  percentuais em 0-100:              {len(percentuais)} colunas OK")

    # --- alerta: feature 100% nula não informa nada e some sem aviso
    vazias = []

    for coluna, _ in ESQUEMA_GOLD:
        if coluna in ("id_aluno", "id_escola"):
            continue


        if df.filter(F.col(coluna).isNotNull()).limit(1).count() == 0:
            vazias.append(coluna)

    logger(f"  colunas 100% nulas:                {len(vazias)}")

    if vazias:
        # Bloqueante, e nao alerta: uma feature sem nenhum valor desaparece
        # do modelo sem erro. Foi assim que a composicao setorial do PIB
        # ficou vazia duas vezes neste projeto — a primeira passou
        # despercebida ate uma consulta no Athena.
        bloqueios.append(f"colunas 100% nulas: {', '.join(vazias)}")

    logger("-" * 60)

    if bloqueios:
        # Derrubar o Job impede que a camada seguinte consuma dado invalido
        raise RuntimeError(
            f"{len(bloqueios)} regra(s) bloqueante(s) reprovada(s): "
            f"{'; '.join(bloqueios)}"
        )

    logger("Todas as regras bloqueantes aprovadas")


def relatar(df: DataFrame, logger) -> None:
    """Registra volume, cobertura e balanceamento do alvo."""

    total = df.count()

    logger("-" * 60)
    logger(f"Linhas:              {total:,}")
    logger(f"Municipios:          {df.select('id_municipio').distinct().count():,}")
    logger(f"Escolas:             {df.select('id_escola').distinct().count():,}")
    logger(f"Colunas:             {len(df.columns)}")

    # Balanceamento do alvo decide a escolha de métrica: com base muito
    # desbalanceada, acurácia deixa de informar.
    positivos = df.filter(F.col("alfabetizado")).count()

    logger("-" * 60)
    logger(
        f"Alfabetizados:       {positivos:,} "
        f"({positivos / total * 100:.1f}%)"
    )
    logger(
        f"Nao alfabetizados:   {total - positivos:,} "
        f"({(total - positivos) / total * 100:.1f}%)"
    )

    anomalos = df.filter(F.col("uf_anomala")).count()

    logger("-" * 60)
    logger(f"Linhas marcadas como UF anomala ({UF_ANOMALA}): {anomalos:,}")

    # Nulo em feature municipal indica municipio ausente de alguma fonte —
    # o join falharia silenciosamente sem esta conferencia.
    sem_contexto = df.filter(
        F.col("mun_taxa_alfabetizacao_adulta").isNull()
    ).count()

    logger(
        f"Sem contexto socioeconomico: {sem_contexto:,} "
        f"({sem_contexto / total * 100:.1f}%)"
    )


def main():
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "BUCKET_DESTINO",
            "DATABASE_SILVER",
            "DATABASE_GOLD",
            "DATABASE_FASE3",
            "ENV",
        ],
    )

    contexto = GlueContext(SparkContext.getOrCreate())

    job = Job(contexto)
    job.init(args["JOB_NAME"], args)

    logger = contexto.get_logger().info

    databases = {
        "silver": args["DATABASE_SILVER"],
        "gold": args["DATABASE_GOLD"],
        "fase3": args["DATABASE_FASE3"],
    }

    logger("=" * 60)
    logger(f"GOLD FASE 3 — features_aluno — ambiente {args['ENV']}")
    logger("=" * 60)
    logger(f"Ano alvo: {ANO_ALVO} · contexto municipal: {ANO_CONTEXTO}")
    logger(f"Colunas de vazamento excluidas: {', '.join(COLUNAS_VAZAMENTO)}")
    logger("")

    fontes = {}

    for apelido, (banco, tabela) in TABELAS.items():
        df = contexto.create_dynamic_frame.from_catalog(
            database=databases[banco], table_name=tabela
        ).toDF()

        fontes[apelido] = df

        logger(f"  {databases[banco]}.{tabela:36} {df.count():>10,} linhas")

    resultado = construir(fontes)

    # Cacheia porque a validacao percorre o DataFrame varias vezes e ele
    # seria recomputado desde os cinco joins a cada passagem.
    resultado.cache()

    validar(resultado, logger)
    relatar(resultado, logger)

    destino = f"s3://{args['BUCKET_DESTINO']}/fase3/{DESTINO}/"

    # Sem coalesce(1): sao milhoes de linhas, e um arquivo unico produziria
    # um grupo de linhas gigante, pior para leitura seletiva. A Fase 2
    # mostrou que coalesce(1) sobre 100 MB aumentou o tamanho em 48%.
    resultado.write.mode("overwrite").parquet(destino)

    logger("")
    logger(f"Gravado: {destino}")

    logger("=" * 60)
    logger("GOLD CONCLUIDA")

    job.commit()


if __name__ == "__main__":
    main()
