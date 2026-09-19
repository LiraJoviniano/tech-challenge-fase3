// ---------------------------------------------------------------------------
// Camada Gold da Fase 3 — features_aluno
//
// A Gold existe para entregar dado pronto para consumo. Como o consumo
// desta fase é um modelo no grão do aluno, a junção hierárquica acontece
// aqui, num Glue Job — não num script de pré-processamento. Feita uma vez,
// validada, e qualquer consumidor futuro recebe a tabela pronta.
//
// É aqui que o Glue se justifica, diferente da ingestão: são 3,87 milhões
// de linhas e cinco joins, volume em que o Spark paga o provisionamento.
// A ingestão de 54 MB continua local, e a assimetria é deliberada.
//
// O Job lê três databases: a Silver e a Gold da Fase 2, mais a Bronze da
// Fase 3. Grava sob `fase3/gold/`, sem tocar nos prefixos da fase
// anterior.
//
// Não há features no grão da escola. O `id_escola` de `fato_aluno` é um
// identificador sintético da avaliação — os 42.497 valores começam todos
// com `60`, prefixo que não corresponde a nenhuma UF —, e o join com
// `fato_escola` retorna zero linhas. O contexto escolar entra agregado por
// município.
//
// As definições espelham ESQUEMA_GOLD em src/transformation/gold_fase3.py.
// Se uma mudar, a outra precisa mudar junto.
// ---------------------------------------------------------------------------

resource "aws_s3_object" "script_gold" {
  bucket = var.bucket
  key    = "fase3/scripts/gold_fase3.py"
  source = var.caminho_script_gold
  etag   = filemd5(var.caminho_script_gold)
}

resource "aws_glue_catalog_database" "gold" {
  name        = "${var.prefixo}_gold"
  description = "Fase 3 - dataset analitico no grao do aluno"

  location_uri = "s3://${var.bucket}/fase3/gold/"

  tags = merge(local.tags_comuns, { Layer = "gold" })
}

resource "aws_glue_job" "gold" {
  name        = "${var.prefixo}_job_gold"
  description = "Constroi features_aluno: aluno x escola x municipio, sem vazamento"
  role_arn    = "arn:aws:iam::${data.aws_caller_identity.atual.account_id}:role/${var.role_glue}"

  glue_version      = "4.0"
  worker_type       = "G.1X"
  number_of_workers = 2

  timeout = 30

  command {
    name            = "glueetl"
    script_location = "s3://${var.bucket}/${aws_s3_object.script_gold.key}"
    python_version  = "3"
  }

  default_arguments = {
    "--BUCKET_DESTINO"  = var.bucket
    "--DATABASE_SILVER" = var.database_silver_fase2
    "--DATABASE_GOLD"   = var.database_gold_fase2
    "--DATABASE_FASE3"  = aws_glue_catalog_database.bronze.name
    "--ENV"             = var.ambiente

    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = "true"
    "--job-language"                     = "python"

    // Le as tabelas pelo Catalog com create_dynamic_frame e spark.sql
    "--enable-glue-datacatalog" = "true"

    // Reconstroi o dataset inteiro a cada execucao
    "--job-bookmark-option" = "job-bookmark-disable"
  }

  tags = merge(local.tags_comuns, { Layer = "gold" })
}

// A conta é descoberta em tempo de execução: o ARN da role varia por
// ambiente, e fixá-lo quebraria em outra conta.
data "aws_caller_identity" "atual" {}

// ---------------------------------------------------------------------------
// Tabela
//
// Schema declarado, não inferido por Crawler. Na Bronze o schema é da
// fonte e descobri-lo é apropriado; aqui ele é decisão: `alfabetizado` é
// boolean porque é o alvo, e as colunas de vazamento simplesmente não
// existem no contrato.
// ---------------------------------------------------------------------------

resource "aws_glue_catalog_table" "features_aluno" {
  name          = "features_aluno"
  database_name = aws_glue_catalog_database.gold.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "parquet"
  }

  storage_descriptor {
    location      = "s3://${var.bucket}/fase3/gold/features_aluno/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      name                  = "parquet"
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"

      parameters = {
        "serialization.format" = "1"
      }
    }

    dynamic "columns" {
      for_each = local.colunas_features_aluno

      content {
        name = columns.value.name
        type = columns.value.type
      }
    }
  }

  depends_on = [aws_glue_job.gold]
}

locals {
  colunas_features_aluno = [
        { name = "ano", type = "int" },
        { name = "id_aluno", type = "string" },
        { name = "id_escola", type = "string" },
        // Chave de agrupamento para GroupKFold, nao feature:
        // 5.570 categorias seriam memorizadas por qualquer codificacao
        { name = "id_municipio", type = "string" },
        { name = "sigla_uf", type = "string" },
        { name = "regiao", type = "string" },
        // Alvo do modelo
        { name = "alfabetizado", type = "boolean" },
        { name = "peso_aluno", type = "double" },
        // RS: anomalia documentada na Fase 2, marcada e nao removida
        { name = "uf_anomala", type = "boolean" },
        { name = "rede_codigo", type = "string" },
        { name = "caderno", type = "string" },
        // Taxa de 2023. A do proprio ano seria a media do alvo dos
        // alunos daquele municipio, e vazaria o resultado da turma
        { name = "mun_taxa_ano_anterior", type = "double" },
        { name = "mun_total_escolas", type = "int" },
        { name = "mun_alunos_por_docente", type = "double" },
        { name = "mun_alunos_por_turma", type = "double" },
        { name = "mun_pct_integral", type = "double" },
        // Contexto escolar entra agregado por municipio: o id_escola da
        // avaliacao e sintetico e nao liga ao Censo Escolar
        { name = "mun_indice_infraestrutura", type = "double" },
        { name = "mun_pct_rural", type = "double" },
        { name = "mun_pct_transporte", type = "double" },
        // Meta pactuada, publicada antes da avaliacao. As demais colunas
        // de trajetoria_meta_2030 derivam de taxa_2024 e seriam vazamento
        { name = "mun_meta_ano_alvo", type = "double" },
        // taxa_2023 - meta_2024: quanto faltava no fim de 2023 para
        // alcancar a meta do ano seguinte
        { name = "mun_distancia_meta_anterior", type = "double" },
        { name = "mun_elegivel_meta", type = "boolean" },
        { name = "mun_populacao", type = "int" },
        { name = "mun_densidade", type = "double" },
        { name = "mun_moradores_por_domicilio", type = "double" },
        { name = "mun_taxa_alfabetizacao_adulta", type = "double" },
        { name = "mun_idade_mediana", type = "double" },
        { name = "mun_indice_envelhecimento", type = "double" },
        { name = "mun_pct_indigena", type = "double" },
        { name = "mun_pct_quilombola", type = "double" },
        { name = "mun_pib_per_capita", type = "double" },
        { name = "mun_pct_va_agropecuaria", type = "double" },
        { name = "mun_pct_va_industria", type = "double" },
        { name = "mun_pct_va_servicos", type = "double" },
        { name = "mun_pct_va_administracao_publica", type = "double" },
  ]
}
