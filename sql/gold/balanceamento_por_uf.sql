-- Taxa de alfabetizacao por UF, no dataset de modelagem.
--
-- Serve de conferencia contra os numeros da Fase 2: se a taxa por UF aqui
-- divergir muito da que a Gold anterior publicou, algum filtro esta
-- cortando alem do previsto.
--
-- O RS aparece separado: a anomalia documentada na Fase 2 desloca a
-- distribuicao inteira do estado, e mistura-lo mascara o diagnostico.

SELECT
    sigla_uf,
    uf_anomala,
    COUNT(*)                                                   AS alunos,
    COUNT(DISTINCT id_municipio)                               AS municipios,
    ROUND(100.0 * SUM(CASE WHEN alfabetizado THEN 1 ELSE 0 END)
          / COUNT(*), 1)                                       AS pct_alfabetizados
FROM features_aluno
GROUP BY sigla_uf, uf_anomala
ORDER BY pct_alfabetizados DESC
