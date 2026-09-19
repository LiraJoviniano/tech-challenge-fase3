"""
Escrita da camada Bronze, com histórico preservado.

Corrige a limitação apontada na avaliação da Fase 2: lá o arquivo era
gravado com nome fixo, e cada execução substituía a anterior no lake. Não
havia como reprocessar uma carga passada nem auditar o que a fonte trazia
em determinada data.

Três mudanças resolvem isso:

    1. Particionamento por data de ingestão, no padrão Hive
       (`dt_ingestao=AAAA-MM-DD`), que o Glue Crawler reconhece
       automaticamente como partição.

       A hierarquia é `<fonte>/<tabela>/dt_ingestao=.../arquivo.parquet`:
       a tabela precisa vir antes da partição. Com duas tabelas de schemas
       distintos dentro da mesma partição, o Crawler desce um nível, cria
       uma tabela por arquivo e `dt_ingestao` deixa de ser chave — foi o
       que aconteceu na primeira tentativa com o Censo.
    2. Coluna `_ingerido_em` em cada registro, com o instante da extração.
    3. Manifesto versionado em `reports/ingestao/`, registrando cada
       execução com volume, custo e caminho.

O caminho segue a convenção Hive, com a **tabela antes da partição**:

    <fonte>_<tabela>/dt_ingestao=AAAA-MM-DD/dados.parquet

A ordem importa. Com a partição acima da tabela, dois schemas distintos
cairiam na mesma pasta de partição e o Glue Crawler tentaria fundi-los
numa tabela só. O nome combina fonte e tabela porque o crawler nomeia a
tabela pela última pasta, e `censo_2022/municipio` colidiria com
`pib_municipal/municipio`.

O manifesto é o que torna a auditoria possível sem depender de listar o S3:
ele vive no repositório e sobrevive ao laboratório.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


# Prefixo próprio da Fase 3. Mantém a separação da Fase 2 sem duplicar
# bucket: os crawlers de lá têm include paths explícitos e não alcançam
# este caminho.
PREFIXO_S3 = "fase3/bronze"

MANIFESTO = settings.raiz / "reports" / "ingestao" / "manifesto.jsonl"

COLUNA_CARIMBO = "_ingerido_em"


def _particao(momento: datetime) -> str:
    """Partição no padrão Hive, reconhecida pelo Glue e pelo Athena."""

    return f"dt_ingestao={momento:%Y-%m-%d}"


def caminho_tabela(fonte: str, tabela: str) -> str:
    """
    Nome da pasta que representa a tabela no lake.

    Combina fonte e tabela porque o Glue Crawler nomeia a tabela pela
    última pasta antes da partição: `censo_2022/municipio` e
    `pib_municipal/municipio` colidiriam.
    """

    return f"{fonte}_{tabela}"


def gravar_local(
    df: pd.DataFrame, fonte: str, tabela: str, momento: datetime
) -> Path:
    """Grava o Parquet particionado, com o carimbo de ingestão."""

    destino = (
        settings.dados_brutos
        / caminho_tabela(fonte, tabela)
        / _particao(momento)
        / "dados.parquet"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)

    # O carimbo vai na linha, não só no caminho: um registro extraído da
    # partição continua sabendo quando foi lido.
    df = df.copy()
    df[COLUNA_CARIMBO] = pd.Timestamp(momento)

    df.to_parquet(destino, index=False)

    logger.info(
        f"  gravado: {destino.relative_to(settings.raiz)} "
        f"({destino.stat().st_size / 1024:.1f} KB)"
    )

    return destino


def enviar_s3(caminho: Path, fonte: str, tabela: str, momento: datetime) -> str:
    """Envia ao S3 preservando a estrutura de partição."""

    chave = (
        f"{PREFIXO_S3}/{caminho_tabela(fonte, tabela)}/"
        f"{_particao(momento)}/dados.parquet"
    )

    boto3.client("s3", region_name=settings.aws_region).upload_file(
        str(caminho), settings.aws_bucket, chave
    )

    destino = f"s3://{settings.aws_bucket}/{chave}"
    logger.info(f"  enviado:  {destino}")

    return destino


def registrar(registro: dict) -> None:
    """
    Acrescenta a execução ao manifesto.

    Formato JSONL: uma linha por execução, sem reescrever o arquivo. Isso
    evita conflito de merge quando duas pessoas ingerem no mesmo dia.
    """

    MANIFESTO.parent.mkdir(parents=True, exist_ok=True)

    with MANIFESTO.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def persistir(
    df: pd.DataFrame,
    fonte: str,
    tabela: str,
    bytes_varridos: int,
    consulta: str,
    momento: datetime | None = None,
    enviar: bool = True,
) -> dict:
    """
    Grava, envia e registra uma extração.

    Devolve o registro do manifesto, para que o script chamador possa
    exibi-lo sem reler o arquivo.
    """

    momento = momento or datetime.now(timezone.utc)

    caminho = gravar_local(df, fonte, tabela, momento)

    destino_s3 = None
    if enviar and settings.aws_bucket:
        try:
            destino_s3 = enviar_s3(caminho, fonte, tabela, momento)
        except Exception as erro:
            # Credencial do laboratório expira a cada sessão. O dado local
            # continua válido, e a falha não deve derrubar a extração.
            logger.warning(f"  S3 indisponivel ({erro}) — dado gravado localmente")

    registro = {
        "momento": momento.isoformat(),
        "fonte": fonte,
        "tabela": tabela,
        "particao": _particao(momento),
        "linhas": len(df),
        "colunas": len(df.columns),
        "bytes_varridos": bytes_varridos,
        "mb_varridos": round(bytes_varridos / 1024**2, 2),
        # as_posix normaliza as barras: o manifesto e versionado e sera
        # lido em Windows, macOS e Linux
        "caminho_local": caminho.relative_to(settings.raiz).as_posix(),
        "caminho_s3": destino_s3,
        "consulta": " ".join(consulta.split()),
    }

    registrar(registro)

    return registro


def ultima_particao(fonte: str, tabela: str) -> Path | None:
    """
    Caminho da carga mais recente de uma tabela.

    Com histórico preservado, ler "o dado atual" deixa de ser trivial — é
    preciso escolher a partição. Esta função centraliza a regra para que
    nenhum consumidor a reimplemente de forma diferente.
    """

    base = settings.dados_brutos / caminho_tabela(fonte, tabela)

    if not base.exists():
        return None

    particoes = sorted(
        p for p in base.glob("dt_ingestao=*") if (p / "dados.parquet").exists()
    )

    if not particoes:
        return None

    return particoes[-1] / "dados.parquet"
