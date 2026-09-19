output "database" {
  description = "Database da Bronze da Fase 3 no Glue Catalog."
  value       = aws_glue_catalog_database.bronze.name
}

output "crawler" {
  description = "Crawler que cataloga as fontes socioeconomicas."
  value       = aws_glue_crawler.bronze.name
}

output "comando_executar_crawler" {
  description = "O Terraform declara o crawler; executa-lo e acao, nao estado."
  value       = "bash infra/executar_crawler.sh"
}

output "consulta_de_verificacao" {
  description = "Prova que a particao por data de ingestao foi reconhecida."
  value       = "SELECT dt_ingestao, COUNT(*) FROM ${aws_glue_catalog_database.bronze.name}.censo_2022_municipio GROUP BY dt_ingestao"
}

output "database_gold" {
  description = "Database da Gold da Fase 3."
  value       = aws_glue_catalog_database.gold.name
}

output "job_gold" {
  description = "Glue Job que constroi features_aluno."
  value       = aws_glue_job.gold.name
}

output "comando_executar_gold" {
  description = "Executar e acao, nao estado."
  value       = "aws glue start-job-run --job-name ${aws_glue_job.gold.name}"
}
