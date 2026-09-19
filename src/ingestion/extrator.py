"""
Extração do BigQuery, com medição antes da execução.

Toda consulta passa por dry run antes de rodar: o BigQuery informa quantos
bytes seriam varridos sem executar nada e sem gerar custo. Medir antes
transformou uma extração de 6,17 GB em 56 MB na Fase 2 — a seleção de
colunas responde por quase toda a economia, porque a cobrança é por coluna
varrida e `LIMIT` corta o retorno, não a varredura.
"""

import logging

from google.cloud import bigquery

from config.settings import settings

logger = logging.getLogger(__name__)


PROJETO_FONTE = "basedosdados"

# Teto por consulta. Acima disso a query falha antes de executar, em vez de
# gerar custo inesperado.
MAX_BYTES_FATURADOS = 2 * 1024**3

# Acima deste volume, a extração é interrompida para revisão do recorte.
LIMITE_ALERTA_GB = 1.0


def cliente() -> bigquery.Client:
    settings.validar()
    return bigquery.Client(project=settings.gcp_project_id)


def medir(sql: str, cli: bigquery.Client | None = None) -> int:
    """Bytes que a consulta varreria, sem executá-la."""

    cli = cli or cliente()

    config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    return cli.query(sql, job_config=config).total_bytes_processed


def extrair(sql: str, cli: bigquery.Client | None = None) -> tuple:
    """
    Mede, confere o limite e extrai.

    Devolve o DataFrame e os bytes varridos, para que o chamador registre
    o consumo no manifesto.
    """

    cli = cli or cliente()

    varredura = medir(sql, cli)
    gb = varredura / 1024**3

    logger.info(f"  varredura estimada: {varredura / 1024**2:.2f} MB")

    if gb > LIMITE_ALERTA_GB:
        raise SystemExit(
            f"Consulta varreria {gb:.2f} GB, acima do limite de "
            f"{LIMITE_ALERTA_GB} GB. Revise a selecao de colunas ou o "
            f"filtro antes de executar."
        )

    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES_FATURADOS
    )

    df = cli.query(sql, job_config=job_config).result().to_dataframe()

    logger.info(f"  registros: {len(df):,}")

    return df, varredura


def montar_sql(
    dataset: str,
    tabela: str,
    colunas: list[str] | None = None,
    filtro: str | None = None,
) -> str:
    """
    Monta a consulta do recorte.

    Colunas explícitas em vez de `SELECT *`: é o que reduz custo de verdade.
    """

    campos = ",\n    ".join(colunas) if colunas else "*"

    sql = f"SELECT\n    {campos}\nFROM `{PROJETO_FONTE}.{dataset}.{tabela}`"

    if filtro:
        sql += f"\nWHERE {filtro}"

    return sql


def valor_unico(sql: str, cli: bigquery.Client | None = None):
    """Primeiro valor da primeira linha, para consultas de metadado."""

    cli = cli or cliente()

    resultado = list(cli.query(sql).result())

    return resultado[0][0] if resultado else None
