"""
Configuração centralizada.

Lê as variáveis do arquivo `.env` na raiz do repositório. Nenhuma credencial
fica no código — o `.env` está no `.gitignore`, e o `.env.example` traz
apenas os nomes das variáveis.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]

load_dotenv(RAIZ / ".env")


@dataclass(frozen=True)
class Configuracao:
    """Parâmetros de ambiente do projeto."""

    # GCP — usado para faturar as consultas ao BigQuery.
    # A Base dos Dados é pública, mas quem executa paga pela varredura.
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "")

    # AWS — origem das camadas Silver e Gold construídas na Fase 2
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_bucket: str = os.getenv("AWS_BUCKET", "")

    ambiente: str = os.getenv("PIPELINE_ENV", "dev")

    # Caminhos locais
    raiz: Path = RAIZ
    dados_brutos: Path = RAIZ / "data" / "raw"
    dados_analiticos: Path = RAIZ / "data" / "analytics"
    amostra: Path = RAIZ / "data" / "sample"

    def validar(self) -> None:
        """Falha cedo e com mensagem clara quando falta configuração."""

        faltando = [
            nome
            for nome, valor in (
                ("GCP_PROJECT_ID", self.gcp_project_id),
                ("AWS_BUCKET", self.aws_bucket),
            )
            if not valor
        ]

        if faltando:
            raise SystemExit(
                f"Variaveis nao configuradas: {', '.join(faltando)}\n"
                f"Copie .env.example para .env e preencha os valores."
            )


settings = Configuracao()
