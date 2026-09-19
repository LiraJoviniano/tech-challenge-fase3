"""
Busca de datasets na Base dos Dados.

Lista os datasets públicos disponíveis e filtra por palavra-chave. Existe
porque presumir o nome de um dataset custa uma execução perdida — e há
centenas deles, com convenções de nomenclatura que variam por órgão.

Uso:
    python src/ingestion/buscar_dataset.py              # lista tudo
    python src/ingestion/buscar_dataset.py cadastro     # filtra
    python src/ingestion/buscar_dataset.py censo ibge   # varios termos
"""

import logging
import sys
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJETO_FONTE = "basedosdados"


def listar(termos: list[str]) -> None:
    """Lista os datasets, filtrando por qualquer um dos termos."""

    settings.validar()
    cli = bigquery.Client(project=settings.gcp_project_id)

    todos = sorted(d.dataset_id for d in cli.list_datasets(PROJETO_FONTE))

    if termos:
        encontrados = [
            d for d in todos if any(t.lower() in d.lower() for t in termos)
        ]
        titulo = f"datasets contendo {' ou '.join(termos)}"
    else:
        encontrados = todos
        titulo = "todos os datasets"

    logger.info("=" * 60)
    logger.info(f"{titulo}: {len(encontrados)} de {len(todos)}")
    logger.info("=" * 60)

    for nome in encontrados:
        logger.info(f"  {nome}")

    if not encontrados:
        logger.info("  nenhum encontrado — tente um termo mais curto")

    logger.info("")
    logger.info("Para explorar um deles:")
    logger.info("    python src/ingestion/explorar_fonte.py <dataset>")


if __name__ == "__main__":
    listar(sys.argv[1:])
