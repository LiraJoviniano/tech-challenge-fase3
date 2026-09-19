-- Historico de ingestao por fonte.
--
-- Prova que a particao por data foi reconhecida pelo Glue e que cargas
-- anteriores coexistem em vez de serem sobrescritas — a correcao da
-- unica critica que a Fase 2 recebeu.
--
-- Com mais de uma particao, qualquer consumidor precisa escolher qual
-- ler. A regra do projeto e a mais recente.

SELECT dt_ingestao, COUNT(*) AS registros
FROM censo_2022_municipio
GROUP BY dt_ingestao
ORDER BY dt_ingestao DESC
