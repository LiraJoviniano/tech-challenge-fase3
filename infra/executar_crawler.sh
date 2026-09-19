#!/usr/bin/env bash
#
# Executa o crawler da Bronze e verifica o resultado.
#
# O crawler é declarado em Terraform; executá-lo é ação, não estado, e por
# isso fica aqui. A verificação no fim confirma que `dt_ingestao` foi
# reconhecida como chave de partição — que é o propósito de catalogar esta
# camada.
#
# Uso:
#   bash infra/executar_crawler.sh
#
# Pré-requisito:
#   cd infra/terraform && terraform apply

set -euo pipefail

# O Git Bash converte caminhos iniciados por barra em caminhos do Windows
# antes de repassar ao AWS CLI. Inofensivo nos demais sistemas.
export MSYS_NO_PATHCONV=1

# Sem isso o AWS CLI abre um paginador interativo e o script parece travar
export AWS_PAGER=""

PREFIXO="${PREFIXO:-fase3}"
REGIAO="${AWS_REGION:-us-east-1}"

DATABASE="${PREFIXO}_bronze"
CRAWLER="${PREFIXO}_crawler_bronze"

INTERVALO=10
MAX_TENTATIVAS=60

info()  { echo "[INFO]  $*"; }
aviso() { echo "[AVISO] $*"; }
erro()  { echo "[ERRO]  $*" >&2; }

separador() { printf '%.0s-' {1..70}; echo; }

verificar_pre_requisitos() {
  if ! aws sts get-caller-identity > /dev/null 2>&1; then
    erro "Credenciais invalidas ou expiradas."
    erro "No AWS Academy, copie novamente em AWS Details > AWS CLI > Show."
    exit 1
  fi

  if ! aws glue get-crawler --name "$CRAWLER" --region "$REGIAO" > /dev/null 2>&1; then
    erro "Crawler ${CRAWLER} nao encontrado."
    erro "Execute antes: cd infra/terraform && terraform apply"
    exit 1
  fi

  info "Crawler ${CRAWLER} encontrado"
}

executar() {
  info "Iniciando execucao..."

  aws glue start-crawler --name "$CRAWLER" --region "$REGIAO"

  local tentativa=0
  local estado=""

  while [ "$tentativa" -lt "$MAX_TENTATIVAS" ]; do
    estado=$(aws glue get-crawler \
      --name "$CRAWLER" --region "$REGIAO" \
      --query 'Crawler.State' --output text)

    if [ "$estado" = "READY" ] && [ "$tentativa" -gt 0 ]; then
      break
    fi

    printf '.'
    sleep "$INTERVALO"
    tentativa=$((tentativa + 1))
  done

  echo

  if [ "$estado" != "READY" ]; then
    erro "Nao concluiu em $((MAX_TENTATIVAS * INTERVALO))s (estado: ${estado})"
    exit 1
  fi

  local resultado
  resultado=$(aws glue get-crawler \
    --name "$CRAWLER" --region "$REGIAO" \
    --query 'Crawler.LastCrawl.Status' --output text)

  info "Concluido com status: ${resultado}"

  if [ "$resultado" != "SUCCEEDED" ]; then
    erro "O crawler nao concluiu com sucesso."
    aws glue get-crawler --name "$CRAWLER" --region "$REGIAO" \
      --query 'Crawler.LastCrawl.ErrorMessage' --output text
    exit 1
  fi
}

listar_tabelas() {
  separador
  info "Tabelas catalogadas em ${DATABASE}:"
  echo

  aws glue get-tables \
    --database-name "$DATABASE" --region "$REGIAO" \
    --query 'TableList[].[Name, length(StorageDescriptor.Columns), length(PartitionKeys)]' \
    --output table
}

verificar_particao() {
  separador
  info "Verificando o reconhecimento da particao por data de ingestao"
  echo

  local tabelas
  tabelas=$(aws glue get-tables \
    --database-name "$DATABASE" --region "$REGIAO" \
    --query 'TableList[].Name' --output text)

  local falhas=0

  for tabela in $tabelas; do
    local chave
    chave=$(aws glue get-table \
      --database-name "$DATABASE" --name "$tabela" --region "$REGIAO" \
      --query 'Table.PartitionKeys[0].Name' --output text 2>/dev/null || echo "None")

    if [ "$chave" = "dt_ingestao" ]; then
      info "  ${tabela}: particionada por ${chave}  OK"
    else
      aviso "  ${tabela}: chave de particao = ${chave}"
      falhas=$((falhas + 1))
    fi
  done

  echo

  if [ "$falhas" -gt 0 ]; then
    aviso "Alguma tabela nao reconheceu dt_ingestao como particao."
    aviso "Confira se o caminho no S3 segue o padrao dt_ingestao=AAAA-MM-DD."
    exit 1
  fi

  info "Historico de ingestao consultavel por data"
}

main() {
  separador
  info "CRAWLER DA BRONZE — FASE 3"
  separador

  verificar_pre_requisitos
  executar
  listar_tabelas
  verificar_particao

  separador
  info "Consulta de verificacao no Athena:"
  echo "    SELECT dt_ingestao, COUNT(*)"
  echo "    FROM ${DATABASE}.censo_2022_municipio"
  echo "    GROUP BY dt_ingestao"
  separador
}

main "$@"
