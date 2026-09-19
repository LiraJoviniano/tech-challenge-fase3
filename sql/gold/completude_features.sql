-- Percentual de nulos em cada feature do modelo.
--
-- Feature com muitos nulos exige decisao antes da modelagem: imputar,
-- descartar ou tratar o nulo como categoria. Feature 100% nula desaparece
-- do modelo sem erro nenhum — foi o que aconteceu com a composicao
-- setorial do PIB na primeira execucao da agregacao.
--
-- Athena nao tem forma direta de iterar colunas, entao a lista e
-- explicita. Ao acrescentar feature nova, acrescente aqui tambem.

SELECT
    'mun_taxa_ano_anterior'          AS feature,
    ROUND(100.0 * SUM(CASE WHEN mun_taxa_ano_anterior IS NULL THEN 1 ELSE 0 END)
          / COUNT(*), 2)             AS pct_nulo
FROM features_aluno
UNION ALL SELECT 'mun_indice_infraestrutura',
    ROUND(100.0 * SUM(CASE WHEN mun_indice_infraestrutura IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM features_aluno
UNION ALL SELECT 'mun_taxa_alfabetizacao_adulta',
    ROUND(100.0 * SUM(CASE WHEN mun_taxa_alfabetizacao_adulta IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM features_aluno
UNION ALL SELECT 'mun_pib_per_capita',
    ROUND(100.0 * SUM(CASE WHEN mun_pib_per_capita IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM features_aluno
UNION ALL SELECT 'mun_pct_va_industria',
    ROUND(100.0 * SUM(CASE WHEN mun_pct_va_industria IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM features_aluno
UNION ALL SELECT 'mun_densidade',
    ROUND(100.0 * SUM(CASE WHEN mun_densidade IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM features_aluno
ORDER BY pct_nulo DESC
