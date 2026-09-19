// ---------------------------------------------------------------------------
// Infraestrutura da Fase 3 — catalogação da Bronze
//
// Escopo deliberadamente pequeno: um database e um crawler. Não há Glue
// Job aqui, e a ausência é decisão, não omissão — transformar 54 MB em
// Spark custaria mais em provisionamento do que a transformação inteira,
// e o processamento local itera mais rápido durante a modelagem.
//
// O que justifica existir: a Fase 2 foi avaliada com uma crítica única —
// a Bronze gravava com nome fixo, sem como reprocessar uma carga passada
// nem auditar o que a fonte trazia em determinada data. A Fase 3 corrigiu
// isso com partição `dt_ingestao=AAAA-MM-DD` no padrão Hive.
//
// Sem catálogo, essa correção é uma pasta no disco. Com o crawler, a
// partição vira chave reconhecida e a consulta no Athena prova que o
// mecanismo funciona:
//
//     SELECT dt_ingestao, COUNT(*)
//     FROM fase3_bronze.censo_2022
//     GROUP BY dt_ingestao
//
// A infraestrutura da Fase 2 permanece intocada: este código lê e grava
// apenas sob o prefixo `fase3/`, e os crawlers de lá têm include paths
// explícitos que não alcançam este caminho.
// ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  // Backend local: o laboratório do AWS Academy descarta os recursos entre
  // sessões, e um state remoto descreveria infraestrutura inexistente.
  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "aws" {
  region = var.regiao

  // Credenciais vêm de ~/.aws/credentials, incluindo o session token do
  // AWS Academy. Nenhuma credencial é declarada aqui.
}

locals {
  tags_comuns = {
    Environment = var.ambiente
    Fase        = "3"
    ManagedBy   = "terraform"
    Project     = "fiap-alfabetizacao"
  }

  // Uma pasta por tabela, no padrão `<fonte>_<tabela>`, sob o prefixo
  // próprio da Fase 3.
  //
  // O nome combina fonte e tabela porque o Crawler batiza a tabela pela
  // última pasta antes da partição: `censo_2022/municipio` e
  // `pib_municipal/municipio` colidiriam, já que as duas fontes têm uma
  // tabela chamada `municipio`.
  //
  // A pasta da tabela precisa vir antes da partição. Com dois schemas
  // distintos dentro da mesma partição, o Crawler desce um nível, cria uma
  // tabela por arquivo e `dt_ingestao` deixa de ser chave — foi o que
  // aconteceu na primeira execução, e a verificação em
  // infra/executar_crawler.sh detectou.
  //
  // Include paths explícitos em vez de apontar para a raiz: varrer o
  // prefixo inteiro deixaria o Crawler decidir se agrupa tabelas de
  // schemas parecidos, e essa decisão é nossa.
  tabelas = [
    "censo_2022_municipio",
    "censo_2022_alfabetizacao_grupo_idade_sexo_raca",
    "pib_municipal_municipio",
  ]
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

resource "aws_glue_catalog_database" "bronze" {
  name        = "${var.prefixo}_bronze"
  description = "Fase 3 - fontes socioeconomicas, particionadas por data de ingestao"

  location_uri = "s3://${var.bucket}/fase3/bronze/"

  tags = merge(local.tags_comuns, { Layer = "bronze" })
}

// ---------------------------------------------------------------------------
// Crawler
// ---------------------------------------------------------------------------

resource "aws_glue_crawler" "bronze" {
  name          = "${var.prefixo}_crawler_bronze"
  description   = "Cataloga as fontes da Fase 3, com dt_ingestao como particao"
  role          = var.role_glue
  database_name = aws_glue_catalog_database.bronze.name

  dynamic "s3_target" {
    for_each = local.tabelas

    content {
      path = "s3://${var.bucket}/fase3/bronze/${s3_target.value}/"
    }
  }

  // Nova partição a cada ingestão. CRAWL_NEW_FOLDERS_ONLY varre apenas o
  // que ainda não foi catalogado, em vez de reprocessar o histórico
  // inteiro — o custo cresceria a cada carga.
  recrawl_policy {
    recrawl_behavior = "CRAWL_NEW_FOLDERS_ONLY"
  }

  // A AWS exige LOG nos dois comportamentos quando o recrawl é
  // CRAWL_NEW_FOLDERS_ONLY, e a restrição é coerente: um crawler que só
  // varre pastas novas não tem base para decidir alterar ou remover o
  // schema de uma tabela existente. Tentar UPDATE_IN_DATABASE aqui é
  // recusado com InvalidInputException.
  //
  // O efeito colateral é desejado neste caso: mudança de schema na fonte
  // fica registrada no log em vez de sobrescrever o que já foi catalogado.
  // Preservar o histórico é o objetivo do desenho.
  schema_change_policy {
    update_behavior = "LOG"
    delete_behavior = "LOG"
  }

  tags = merge(local.tags_comuns, { Layer = "bronze" })
}
