"""
Ingestão do PIB municipal — capacidade econômica do território.

Complementa o Censo: um descreve a condição das famílias, o outro a
atividade econômica do município. A série tem cerca de 22 anos; o recorte
extrai apenas os anos recentes, suficientes para contexto e comparação.

O ano mais recente é descoberto em tempo de execução. A série do PIB
municipal costuma ter dois a três anos de defasagem em relação ao ano
corrente, e fixar o valor no código quebraria silenciosamente quando o
IBGE publicasse a edição seguinte.

Uso:
    python src/ingestion/ingestao_pib.py
    python src/ingestion/ingestao_pib.py --medir
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion import extrator, writer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


FONTE = "pib_municipal"
DATASET = "br_ibge_pib"
TABELA = "municipio"

# Quantos anos trazer a partir do mais recente. Três permitem calcular
# variação e suavizar oscilação pontual — um município com uma obra grande
# num ano específico teria PIB atípico.
ANOS_RECENTES = 3

COLUNAS = [
    "id_municipio",
    "ano",
    "pib",
    "va_agropecuaria",
    "va_industria",
    "va_servicos",
    "va_adespss",
]


def ano_mais_recente(cli) -> int:
    """Descobre o último ano da série, em vez de presumi-lo."""

    sql = f"SELECT MAX(ano) FROM `{extrator.PROJETO_FONTE}.{DATASET}.{TABELA}`"

    ano = extrator.valor_unico(sql, cli)

    logger.info(f"Ano mais recente da serie: {ano}")

    return int(ano)


def main():
    apenas_medir = "--medir" in sys.argv

    logger.info("=" * 70)
    logger.info("PIB MUNICIPAL — INGESTAO PARA A BRONZE")
    logger.info("=" * 70)

    cli = extrator.cliente()

    ultimo = ano_mais_recente(cli)
    primeiro = ultimo - ANOS_RECENTES + 1

    logger.info(f"Recorte: {primeiro} a {ultimo}")
    logger.info("")

    sql = extrator.montar_sql(
        DATASET, TABELA, COLUNAS, f"ano BETWEEN {primeiro} AND {ultimo}"
    )

    if apenas_medir:
        varredura = extrator.medir(sql, cli)
        logger.info(f"  varredura: {varredura / 1024**2:.2f} MB")
        logger.info("")
        logger.info("Modo medicao: nada foi extraido nem gravado.")
        return

    df, varredura = extrator.extrair(sql, cli)

    municipios = df["id_municipio"].nunique()
    anos = sorted(df["ano"].unique().tolist())

    logger.info(f"  municipios distintos: {municipios:,}")
    logger.info(f"  anos: {anos}")

    # PIB absoluto faria o modelo aprender tamanho de município. A conversão
    # em per capita acontece na Gold, onde a população do Censo está
    # disponível — a Bronze permanece fiel à origem.
    registro = writer.persistir(
        df=df,
        fonte=FONTE,
        tabela=TABELA,
        bytes_varridos=varredura,
        consulta=sql,
    )

    logger.info(f"  particao: {registro['particao']}")
    logger.info("")
    logger.info("-" * 70)
    logger.info("Manifesto: reports/ingestao/manifesto.jsonl")


if __name__ == "__main__":
    main()
