-- Verificacao geral do dataset analitico.
--
-- Primeira consulta a rodar depois de cada execucao do Job. Confere
-- volume, cobertura, balanceamento do alvo e completude dos joins.
--
-- O balanceamento decide a escolha de metrica: com base muito
-- desequilibrada, acuracia deixa de informar e recall passa a valer mais.
--
-- Nulo em mun_taxa_alfabetizacao_adulta indica join que nao encontrou par
-- — e um join incompleto some sem erro.
--
-- Nao ha features no grao da escola: o id_escola da avaliacao e sintetico
-- e nao liga ao Censo Escolar. O contexto escolar entra agregado por
-- municipio.

SELECT
    COUNT(*)                                                   AS linhas,
    COUNT(DISTINCT id_municipio)                               AS municipios,
    COUNT(DISTINCT id_escola)                                  AS escolas,
    SUM(CASE WHEN alfabetizado THEN 1 ELSE 0 END)              AS alfabetizados,
    ROUND(100.0 * SUM(CASE WHEN alfabetizado THEN 1 ELSE 0 END)
          / COUNT(*), 1)                                       AS pct_alfabetizados,
    SUM(CASE WHEN mun_taxa_alfabetizacao_adulta IS NULL
             THEN 1 ELSE 0 END)                                AS sem_contexto_municipio,
    SUM(CASE WHEN uf_anomala THEN 1 ELSE 0 END)                AS linhas_rs
FROM features_aluno
