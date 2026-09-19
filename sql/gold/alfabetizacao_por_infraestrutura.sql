-- Relacao entre infraestrutura escolar do municipio e alfabetizacao.
--
-- O indice de infraestrutura e agregado por municipio, ponderado por
-- matricula, e vem da Gold da Fase 2. Nao ha versao no grao da escola: o
-- id_escola da avaliacao e sintetico e nao liga ao Censo Escolar.
--
-- Leitura descritiva, nao causal. Municipio com escolas melhor equipadas
-- ter taxa maior nao prova que equipamento alfabetiza — pode indicar
-- municipio com mais recurso, que tambem investe em outras coisas.
--
-- Exclui o RS pela anomalia documentada na Fase 2.

WITH faixas AS (
    SELECT
        alfabetizado,
        NTILE(4) OVER (ORDER BY mun_indice_infraestrutura) AS quartil,
        mun_indice_infraestrutura,
        mun_alunos_por_docente
    FROM features_aluno
    WHERE NOT uf_anomala
      AND mun_indice_infraestrutura IS NOT NULL
)
SELECT
    quartil,
    COUNT(*)                                                   AS alunos,
    ROUND(AVG(mun_indice_infraestrutura), 1)                   AS indice_medio,
    ROUND(AVG(mun_alunos_por_docente), 1)                      AS alunos_por_docente,
    ROUND(100.0 * SUM(CASE WHEN alfabetizado THEN 1 ELSE 0 END)
          / COUNT(*), 1)                                       AS pct_alfabetizados
FROM faixas
GROUP BY quartil
ORDER BY quartil
