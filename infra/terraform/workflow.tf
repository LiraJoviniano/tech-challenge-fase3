// ---------------------------------------------------------------------------
// Orquestração — Glue Workflow
//
// Encadeia catalogação e construção da Gold dentro da AWS:
//
//     trigger sob demanda
//       -> crawler da Bronze
//          -> (SUCCEEDED) job da Gold
//
// A Gold só é gerada se a catalogação tiver sucesso. Sem isso, o Job leria
// um Catalog desatualizado e produziria dado silenciosamente errado — a
// partição nova não estaria registrada, e a leitura traria a carga
// anterior sem qualquer aviso.
//
// **A ingestão fica fora do fluxo, e isso é limitação declarada.** Glue
// Workflow encadeia apenas crawlers e jobs; os dois scripts de ingestão
// são Python local, executados à mão antes de iniciar o fluxo.
//
// Convertê-los em Glue Jobs Python Shell traria o ciclo inteiro para
// dentro do Workflow, e foi avaliado: o custo é baixo, 0,0625 DPU, mas as
// ingestões somam 54 MB e rodam em segundos localmente. Empacotá-las em
// Glue para ganhar orquestração de duas etapas que executam uma vez por
// ano não se justifica.
// ---------------------------------------------------------------------------

resource "aws_glue_workflow" "fase3" {
  name        = "${var.prefixo}_workflow"
  description = "Cataloga a Bronze e constroi a Gold no grao do aluno"

  // Disponibiliza os parâmetros para todas as etapas, evitando repetir
  // bucket e database em cada uma.
  default_run_properties = {
    bucket          = var.bucket
    database_bronze = aws_glue_catalog_database.bronze.name
    database_gold   = aws_glue_catalog_database.gold.name
  }

  tags = merge(local.tags_comuns, { Layer = "orquestracao" })
}

// Início do fluxo.
//
// ON_DEMAND em vez de SCHEDULED: o Censo Demográfico é decenal, o PIB é
// anual e a avaliação do INEP é anual. Agendamento diário dispararia
// execuções reprocessando o mesmo dado. Para agendar, troque o type por
// SCHEDULED e informe uma expressão cron.
resource "aws_glue_trigger" "inicio" {
  name          = "${var.prefixo}_trigger_inicio"
  description   = "Inicia o fluxo catalogando a Bronze"
  type          = "ON_DEMAND"
  workflow_name = aws_glue_workflow.fase3.name

  actions {
    crawler_name = aws_glue_crawler.bronze.name
  }

  tags = local.tags_comuns
}

resource "aws_glue_trigger" "bronze_para_gold" {
  name          = "${var.prefixo}_trigger_gold"
  description   = "Constroi a Gold quando a Bronze estiver catalogada"
  type          = "CONDITIONAL"
  workflow_name = aws_glue_workflow.fase3.name

  predicate {
    conditions {
      crawler_name = aws_glue_crawler.bronze.name
      crawl_state  = "SUCCEEDED"
    }
  }

  actions {
    job_name = aws_glue_job.gold.name
  }

  tags = local.tags_comuns
}
