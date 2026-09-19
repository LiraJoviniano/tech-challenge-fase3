#!/usr/bin/env bash
#
# Dispara o Workflow e acompanha até o fim.
#
# O Terraform declara o fluxo; iniciar a execução é ação, e por isso fica
# aqui. Substitui a sequência manual de executar_crawler.sh seguido de
# start-job-run.
#
# A ingestão continua fora do fluxo — Glue Workflow encadeia apenas
# crawlers e jobs, e os scripts de ingestão são Python local. Rode-os
# antes, se houver carga nova:
#
#     python src/ingestion/ingestao_censo_2022.py
#     python src/ingestion/ingestao_pib.py
#
# Uso:
#   bash infra/executar_workflow.sh
#
# Pré-requisito:
#   cd infra/terraform && terraform apply

set -euo pipefail

export MSYS_NO_PATHCONV=1

# Sem isso o AWS CLI abre um paginador interativo e o script parece travar
export AWS_PAGER=""

PREFIXO="${PREFIXO:-fase3}"
BUCKET="${BUCKET:-fiap-ai-scientist-fase-02}"
REGIAO="${AWS_REGION:-us-east-1}"

WORKFLOW="${PREFIXO}_workflow"
DATABASE_GOLD="${PREFIXO}_gold"

INTERVALO=20
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

  if ! aws glue get-workflow --name "$WORKFLOW" --region "$REGIAO" > /dev/null 2>&1; then
    erro "Workflow ${WORKFLOW} nao encontrado."
    erro "Execute antes: cd infra/terraform && terraform apply"
    exit 1
  fi

  info "Workflow ${WORKFLOW} encontrado"

  # O Glue recusa duas execucoes simultaneas do mesmo Job, e a segunda
  # falha em zero segundo com "Max concurrent runs exceeded". Sem esta
  # verificacao, o fluxo inteiro e marcado como falho por uma corrida —
  # e o erro nao diz que basta esperar.
  local em_execucao
  em_execucao=$(aws glue get-job-runs \
    --job-name "${PREFIXO}_job_gold" --region "$REGIAO" \
    --query 'length(JobRuns[?JobRunState==`RUNNING`])' \
    --output text 2>/dev/null || echo "0")

  if [ "$em_execucao" != "0" ]; then
    erro "Ja existe execucao do Job em andamento."
    erro "Aguarde a conclusao antes de disparar o fluxo:"
    erro ""
    erro "    aws glue get-job-runs --job-name ${PREFIXO}_job_gold \\"
    erro "      --max-items 1 --query 'JobRuns[0].JobRunState' --output text"
    exit 1
  fi

  local crawler_estado
  crawler_estado=$(aws glue get-crawler \
    --name "${PREFIXO}_crawler_bronze" --region "$REGIAO" \
    --query 'Crawler.State' --output text 2>/dev/null || echo "READY")

  if [ "$crawler_estado" != "READY" ]; then
    erro "Crawler em estado ${crawler_estado}. Aguarde ficar READY."
    exit 1
  fi
}

disparar() {
  info "Iniciando execucao do fluxo..."

  EXECUCAO=$(aws glue start-workflow-run \
    --name "$WORKFLOW" --region "$REGIAO" \
    --query 'RunId' --output text)

  info "RunId: ${EXECUCAO}"
}

acompanhar() {
  local tentativa=0
  local estado=""
  local anterior=""

  while [ "$tentativa" -lt "$MAX_TENTATIVAS" ]; do
    estado=$(aws glue get-workflow-run \
      --name "$WORKFLOW" --run-id "$EXECUCAO" --region "$REGIAO" \
      --query 'Run.Status' --output text)

    # Progresso por etapa, para nao ficar so pontinhos por varios minutos
    local resumo
    resumo=$(aws glue get-workflow-run \
      --name "$WORKFLOW" --run-id "$EXECUCAO" --region "$REGIAO" \
      --query 'Run.Statistics.[SucceededActions,FailedActions,RunningActions,TotalActions]' \
      --output text 2>/dev/null || echo "")

    if [ "$resumo" != "$anterior" ] && [ -n "$resumo" ]; then
      echo
      info "Etapas — concluidas/falhas/executando/total: ${resumo//	//}"
      anterior="$resumo"
    fi

    case "$estado" in
      COMPLETED|STOPPED|ERROR) break ;;
    esac

    printf '.'
    sleep "$INTERVALO"
    tentativa=$((tentativa + 1))
  done

  echo

  if [ "$estado" != "COMPLETED" ]; then
    erro "Fluxo terminou como ${estado}"
    detalhar_etapas
    exit 1
  fi

  info "Fluxo concluido"
}

detalhar_etapas() {
  separador
  info "Situacao de cada etapa:"
  echo

  aws glue get-workflow-run \
    --name "$WORKFLOW" --run-id "$EXECUCAO" --region "$REGIAO" \
    --include-graph \
    --query 'Run.Graph.Nodes[].[Type, Name, JobDetails.JobRuns[0].JobRunState, CrawlerDetails.Crawls[0].State]' \
    --output table 2>/dev/null || aviso "Detalhe do grafo indisponivel"
}

verificar_saida() {
  separador
  info "Camada Gold no S3:"
  echo

  aws s3 ls "s3://${BUCKET}/fase3/gold/features_aluno/" \
    --recursive --human-readable --summarize --region "$REGIAO" | tail -8

  separador
  info "Para conferir o conteudo:"
  echo "    bash scripts/consultar.sh verificacao_features_aluno"
}

main() {
  separador
  info "WORKFLOW — FASE 3"
  info "crawler da Bronze -> job da Gold"
  separador

  verificar_pre_requisitos
  disparar
  acompanhar
  detalhar_etapas
  verificar_saida

  separador
}

main "$@"
