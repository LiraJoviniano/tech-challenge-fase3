"""
Exploração de datasets do BigQuery, antes de qualquer extração.

Lista as tabelas de um dataset, mede o volume por dry run e mostra as
colunas — tudo sem executar consulta e sem gerar custo.

Uso:
    python src/ingestion/explorar_fonte.py br_mds_cadastro_unico
    python src/ingestion/explorar_fonte.py br_ibge_censo_2022
    python src/ingestion/explorar_fonte.py <dataset> <tabela>
"""

import logging
import sys
from pathlib import Path

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


PROJETO_FONTE = "basedosdados"

# Acima deste volume, extrair a tabela inteira é inviável dentro da cota
# diária configurada no projeto GCP.
LIMITE_GB_SEGURO = 5

# Termos que sinalizam colunas de interesse para o grão municipal.
# A busca é por substring: o objetivo é descobrir o que existe.
TERMOS_INTERESSE = [
    "municipio",
    "uf",
    "ano",
    "mes",
    "data",
    "populacao",
    "domicilio",
    "familia",
    "pessoa",
    "renda",
    "pobreza",
    "extrema",
    "beneficio",
    "cadastrada",
    "idade",
    "alfabetiz",
    "escolaridade",
    "saneamento",
    "agua",
    "esgoto",
    "energia",
]


def cliente() -> bigquery.Client:
    settings.validar()
    return bigquery.Client(project=settings.gcp_project_id)


def estimar(cli: bigquery.Client, dataset: str, tabela: str) -> dict:
    """Mede volume e colunas sem executar consulta."""

    caminho = f"{PROJETO_FONTE}.{dataset}.{tabela}"

    meta = cli.get_table(caminho)

    config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = cli.query(f"SELECT * FROM `{caminho}`", job_config=config)

    return {
        "tabela": tabela,
        "linhas": meta.num_rows,
        "colunas": len(meta.schema),
        "gb": job.total_bytes_processed / 1024**3,
        "campos": [(c.name, c.field_type) for c in meta.schema],
        "particionada": meta.time_partitioning is not None
        or meta.range_partitioning is not None,
    }


def visao_geral(dataset: str) -> None:
    """Panorama do dataset, com alerta para tabelas grandes."""

    cli = cliente()

    logger.info("=" * 78)
    logger.info(f"{PROJETO_FONTE}.{dataset}")
    logger.info("=" * 78)

    try:
        tabelas = sorted(
            t.table_id for t in cli.list_tables(f"{PROJETO_FONTE}.{dataset}")
        )
    except NotFound:
        # Nome de dataset errado e o erro mais comum aqui: sao centenas
        # deles, com convencoes que variam por orgao.
        logger.error(f"Dataset `{dataset}` nao existe em {PROJETO_FONTE}.")
        logger.error("")
        logger.error("Para descobrir o nome correto:")
        logger.error("    python src/ingestion/buscar_dataset.py <termo>")
        return

    if not tabelas:
        logger.error("Dataset existe, mas nao tem tabelas visiveis.")
        return

    for nome in tabelas:
        try:
            info = estimar(cli, dataset, nome)
        except Exception as erro:
            logger.error(f"{nome}: {erro}")
            continue

        alerta = "  <-- GRANDE" if info["gb"] > LIMITE_GB_SEGURO else ""
        parte = " [particionada]" if info["particionada"] else ""

        logger.info(
            f"{info['tabela']:34} {info['linhas']:>14,} linhas  "
            f"{info['colunas']:>4} colunas  {info['gb']:>9.2f} GB"
            f"{parte}{alerta}"
        )

    logger.info("-" * 78)
    logger.info(
        f"Tabelas acima de {LIMITE_GB_SEGURO} GB nao devem ser extraidas inteiras."
    )
    logger.info("Para detalhar uma tabela:")
    logger.info(f"    python src/ingestion/explorar_fonte.py {dataset} <tabela>")


def detalhar(dataset: str, tabela: str) -> None:
    """Mostra as colunas de uma tabela, destacando as de interesse."""

    cli = cliente()
    info = estimar(cli, dataset, tabela)

    logger.info("=" * 78)
    logger.info(
        f"{dataset}.{tabela} — {info['linhas']:,} linhas, "
        f"{info['colunas']} colunas, {info['gb']:.2f} GB"
    )
    logger.info("=" * 78)

    relevantes = [
        (n, t)
        for n, t in info["campos"]
        if any(termo in n.lower() for termo in TERMOS_INTERESSE)
    ]

    outras = [(n, t) for n, t in info["campos"] if (n, t) not in relevantes]

    logger.info(f"\nColunas de interesse ({len(relevantes)}):\n")
    for nome, tipo in relevantes:
        logger.info(f"  {nome:44} {tipo}")

    if outras:
        logger.info(f"\nDemais colunas ({len(outras)}):\n")
        for nome, tipo in outras:
            logger.info(f"  {nome:44} {tipo}")

    logger.info("")
    logger.info("-" * 78)
    logger.info("Antes de extrair, meca o recorte real: selecao de colunas e")
    logger.info("filtro de ano sao o que reduz custo. O BigQuery cobra por")
    logger.info("coluna varrida, e LIMIT corta o retorno, nao a varredura.")


def main():
    if len(sys.argv) < 2:
        logger.info("Uso:")
        logger.info("    python src/ingestion/explorar_fonte.py <dataset>")
        logger.info("    python src/ingestion/explorar_fonte.py <dataset> <tabela>")
        logger.info("")
        logger.info("Candidatos para a Fase 3:")
        logger.info("    br_mds_cadastro_unico    vulnerabilidade social, mensal")
        logger.info("    br_ibge_censo_2022       populacao e domicilios")
        raise SystemExit(1)

    if len(sys.argv) == 2:
        visao_geral(sys.argv[1])
    else:
        detalhar(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
