"""
Ingestão do Censo Demográfico 2022 — contexto socioeconômico municipal.

Duas tabelas, ambas no grão do município ou agregáveis a ele:

    municipio                              população, domicílios, área e
                                           alfabetização adulta
    alfabetizacao_grupo_idade_sexo_raca    desagregação por faixa etária,
                                           sexo e cor/raça

A faixa etária da segunda começa em 15 anos — nenhuma criança avaliada
pelo indicador do 2º ano aparece nela. Não há vazamento: é escolaridade
adulta, preditor de desempenho escolar infantil.

Uso:
    python src/ingestion/ingestao_censo_2022.py
    python src/ingestion/ingestao_censo_2022.py --medir   # so mede, nao extrai
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


FONTE = "censo_2022"
DATASET = "br_ibge_censo_2022"

# `taxa_alfabetizacao` aqui é da população de 15 anos ou mais, medida pelo
# Censo — fenômeno diferente do indicador do 2º ano do INEP, que já existe
# na Gold da Fase 2 com o mesmo nome. Renomear na origem evita que alguém
# confunda as duas depois do join.
RECORTE = {
    "municipio": {
        "colunas": [
            "id_municipio",
            "sigla_uf",
            "populacao",
            "domicilios",
            "area",
            "taxa_alfabetizacao AS taxa_alfabetizacao_adulta_censo",
            "idade_mediana",
            "indice_envelhecimento",
            "populacao_indigena",
            "populacao_quilombola",
        ],
        "filtro": None,
    },
    "alfabetizacao_grupo_idade_sexo_raca": {
        "colunas": [
            "id_municipio",
            "grupo_idade",
            "sexo",
            "cor_raca",
            "alfabetizacao",
            "populacao",
        ],
        "filtro": None,
    },
}


def main():
    apenas_medir = "--medir" in sys.argv

    logger.info("=" * 70)
    logger.info("CENSO DEMOGRAFICO 2022 — INGESTAO PARA A BRONZE")
    logger.info("=" * 70)

    cli = extrator.cliente()
    total_varrido = 0

    for tabela, recorte in RECORTE.items():
        logger.info("")
        logger.info(f"{tabela}")

        sql = extrator.montar_sql(
            DATASET, tabela, recorte["colunas"], recorte["filtro"]
        )

        if apenas_medir:
            varredura = extrator.medir(sql, cli)
            total_varrido += varredura
            logger.info(f"  varredura: {varredura / 1024**2:.2f} MB")
            continue

        df, varredura = extrator.extrair(sql, cli)
        total_varrido += varredura

        registro = writer.persistir(
            df=df,
            fonte=FONTE,
            tabela=tabela,
            bytes_varridos=varredura,
            consulta=sql,
        )

        logger.info(f"  particao: {registro['particao']}")

    logger.info("")
    logger.info("-" * 70)
    logger.info(f"Total varrido: {total_varrido / 1024**2:.2f} MB")

    if apenas_medir:
        logger.info("Modo medicao: nada foi extraido nem gravado.")
    else:
        logger.info("Manifesto: reports/ingestao/manifesto.jsonl")


if __name__ == "__main__":
    main()
