-- Quantos alunos por municipio, e o que isso significa para a validacao.
--
-- As features municipais se repetem em todos os alunos do municipio. O
-- dataset tem milhoes de linhas, mas o tamanho efetivo da amostra para
-- variaveis municipais e o numero de municipios.
--
-- Consequencia direta: um train_test_split aleatorio coloca alunos do
-- mesmo municipio em treino e teste, e o modelo memoriza o contexto sem
-- generalizar. E vazamento sem nenhuma coluna proibida envolvida.
--
-- A correcao e GroupShuffleSplit e GroupKFold agrupando por id_municipio.

SELECT
    COUNT(*)                                   AS linhas,
    COUNT(DISTINCT id_municipio)               AS municipios,
    ROUND(1.0 * COUNT(*) / COUNT(DISTINCT id_municipio), 1) AS alunos_por_municipio,
    MIN(alunos_no_municipio)                   AS minimo,
    APPROX_PERCENTILE(alunos_no_municipio, 0.5) AS mediana,
    MAX(alunos_no_municipio)                   AS maximo
FROM (
    SELECT
        id_municipio,
        COUNT(*) OVER (PARTITION BY id_municipio) AS alunos_no_municipio
    FROM features_aluno
)
