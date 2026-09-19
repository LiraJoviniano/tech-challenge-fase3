"""
Sincronização das camadas da Fase 2.

Baixa as tabelas da Silver e da Gold construídas no Tech Challenge
anterior, que são a base de features deste projeto. A Fase 3 consome e não
modifica: nada é gravado de volta nos prefixos da Fase 2.

Por que baixar em vez de consultar por Athena a cada uso: o ambiente do
AWS Academy expira e apaga os recursos entre sessões. Com o dado em disco,
a modelagem continua possível depois que o laboratório morrer — e o mesmo
vale para quem clonar o repositório e não tiver acesso à conta.

Uso:
    python src/ingestion/sincronizar_fase2.py
    python src/ingestion/sincronizar_fase2.py --listar   # so mostra o que existe
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402
from ingestion import writer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


FONTE = "fase2"

# Tabelas necessárias para o dataset analítico no grão do aluno.
#
# `fato_aluno` está na Silver, não na Gold: o enunciado pede que os dados
# venham da Gold, mas ela é agregada por município e o alvo do modelo é
# individual. A escolha está registrada no README.
TABELAS = {
    # camada  # caminho no S3                              # papel
    "silver/fatos/fato_aluno": "alvo e atributos do estudante",
    "silver/fatos/fato_escola": "infraestrutura e corpo docente",
    "silver/dimensoes/dim_territorio": "territorio, 5.570 municipios",
    "gold/analiticos/features_municipio": "contexto municipal e alvos agregados",
    "gold/indicadores/trajetoria_meta_2030": "situacao do municipio frente a meta",
}


def cliente():
    settings.validar()
    return boto3.client("s3", region_name=settings.aws_region)


def listar_objetos(s3, prefixo: str) -> list:
    """Lista os Parquet sob um prefixo."""

    resposta = s3.list_objects_v2(Bucket=settings.aws_bucket, Prefix=prefixo)

    return [
        item
        for item in resposta.get("Contents", [])
        if item["Key"].endswith(".parquet")
    ]


def nome_tabela(prefixo: str) -> str:
    """Último segmento do caminho vira o nome local da tabela."""

    return prefixo.rstrip("/").split("/")[-1]


def listar() -> None:
    """Mostra o que existe no bucket, sem baixar."""

    s3 = cliente()

    logger.info("=" * 70)
    logger.info(f"CAMADAS DA FASE 2 EM s3://{settings.aws_bucket}")
    logger.info("=" * 70)

    for prefixo, papel in TABELAS.items():
        objetos = listar_objetos(s3, prefixo)

        if not objetos:
            logger.warning(f"  {nome_tabela(prefixo):26} AUSENTE  — {prefixo}")
            continue

        tamanho = sum(o["Size"] for o in objetos) / 1024**2

        logger.info(
            f"  {nome_tabela(prefixo):26} {len(objetos):>2} arquivo(s)  "
            f"{tamanho:>8.1f} MB  — {papel}"
        )


def sincronizar() -> None:
    """Baixa as tabelas, registrando a operação no manifesto."""

    s3 = cliente()
    momento = datetime.now(timezone.utc)

    logger.info("=" * 70)
    logger.info("SINCRONIZACAO DAS CAMADAS DA FASE 2")
    logger.info("=" * 70)

    ausentes = []
    total_mb = 0.0

    for prefixo, papel in TABELAS.items():
        tabela = nome_tabela(prefixo)
        objetos = listar_objetos(s3, prefixo)

        if not objetos:
            logger.error(f"  {tabela:26} AUSENTE em {prefixo}")
            ausentes.append(tabela)
            continue

        # Mesma hierarquia da ingestao: <fonte>_<tabela>/particao. Manter
        # um padrao unico evita que cada consumidor precise saber de onde
        # o dado veio para montar o caminho.
        destino = (
            settings.dados_brutos
            / writer.caminho_tabela(FONTE, tabela)
            / f"dt_ingestao={momento:%Y-%m-%d}"
        )
        destino.mkdir(parents=True, exist_ok=True)

        baixados = 0
        bytes_tabela = 0

        for objeto in objetos:
            arquivo = destino / Path(objeto["Key"]).name

            s3.download_file(settings.aws_bucket, objeto["Key"], str(arquivo))

            baixados += 1
            bytes_tabela += objeto["Size"]

        mb = bytes_tabela / 1024**2
        total_mb += mb

        logger.info(f"  {tabela:26} {baixados:>2} arquivo(s)  {mb:>8.1f} MB")

        writer.registrar(
            {
                "momento": momento.isoformat(),
                "fonte": FONTE,
                "tabela": tabela,
                "particao": f"dt_ingestao={momento:%Y-%m-%d}",
                "arquivos": baixados,
                "bytes": bytes_tabela,
                "mb": round(mb, 2),
                "origem_s3": f"s3://{settings.aws_bucket}/{prefixo}",
                "caminho_local": destino.relative_to(settings.raiz).as_posix(),
                "papel": papel,
            }
        )

    logger.info("-" * 70)
    logger.info(f"Total: {total_mb:.1f} MB")

    if ausentes:
        logger.error("")
        logger.error(f"Tabelas ausentes: {', '.join(ausentes)}")
        logger.error(
            "O ambiente do laboratorio pode ter sido reiniciado. "
            "Reconstrua a Fase 2 antes de prosseguir."
        )
        raise SystemExit(1)

    logger.info("Manifesto: reports/ingestao/manifesto.jsonl")


def main():
    if "--listar" in sys.argv:
        listar()
        return

    sincronizar()


if __name__ == "__main__":
    main()
