variable "prefixo" {
  description = "Prefixo dos recursos. Separa a Fase 3 da infraestrutura da Fase 2 na mesma conta."
  type        = string
  default     = "fase3"

  validation {
    condition     = can(regex("^[a-z][a-z0-9_]*$", var.prefixo))
    error_message = "O prefixo deve ser minusculo e conter apenas letras, numeros e underscore."
  }
}

variable "bucket" {
  description = "Bucket do data lake, compartilhado com a Fase 2."
  type        = string
  default     = "fiap-ai-scientist-fase-02"
}

variable "regiao" {
  description = "Regiao AWS. O AWS Academy libera apenas us-east-1."
  type        = string
  default     = "us-east-1"
}

variable "role_glue" {
  description = "Role assumida pelo Crawler. No AWS Academy nao e possivel criar roles; a LabRole ja vem provisionada."
  type        = string
  default     = "LabRole"
}

variable "ambiente" {
  description = "Ambiente de execucao: dev ou prod."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "prod"], var.ambiente)
    error_message = "O ambiente deve ser dev ou prod."
  }
}

variable "caminho_script_gold" {
  description = "Caminho local do script PySpark da Gold, relativo a infra/terraform."
  type        = string
  default     = "../../src/transformation/gold_fase3.py"
}

variable "database_silver_fase2" {
  description = "Database da Silver da Fase 2, lido sem alteracao."
  type        = string
  default     = "alfabetizacao_silver"
}

variable "database_gold_fase2" {
  description = "Database da Gold da Fase 2, lido sem alteracao."
  type        = string
  default     = "alfabetizacao_gold"
}
