"""Interpretação do modelo escolhido — o que ele usa, e o que o contexto explica.

O HistGradientBoosting foi o modelo escolhido pelo grupo. Interpretá-lo exige separar duas
perguntas que o SHAP do notebook 05 mistura:

1. **O que o modelo usa para prever?** Responde-se com o SHAP do modelo completo. O resultado é
   dominado por `mun_taxa_ano_anterior` e `mun_distancia_meta_anterior` — que carregam quase a
   mesma informação, porque a distância é a própria taxa de 2023 menos a meta. O município que
   alfabetizou bem no ano anterior tende a alfabetizar bem no seguinte.

2. **Quais fatores de contexto se associam à alfabetização?** É a pergunta do enunciado, e a
   primeira resposta não serve a ela: persistência não é alavanca de política. Responde-se
   retirando as duas variáveis de histórico e vendo o que o modelo passa a usar. Se o contexto
   socioeconômico só aparece nesse cenário, é porque já estava contido no resultado do ano
   anterior.

**Os dois modelos são treinados sem o RS.** O modelo final do grupo, treinado com o estado,
aprendeu `sigla_uf_RS` como a terceira variável mais importante: a queda uniforme de 2024 é uma
alteração na origem do dado, documentada na Fase 2, e o modelo a tomou por fator educacional.
Retirá-lo do treino separa o que é ensino do que é medição.

O SHAP do HistGradientBoosting vem em log-odds da classe de risco: valor positivo aumenta a
chance de a criança não se alfabetizar.

Uso: python -m src.evaluation.interpretacao
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.avaliacao import PASTA_RELATORIOS, calcular_shap, treinar_e_avaliar
from src.modeling.modelos import limitar_por_municipio
from src.preprocessing.pipeline import carregar_decisoes, preparar_dados
from src.visualization.graficos import COR, _salvar, plot_shap

NOME = "hist_gradient_boosting"

# A distância à meta é a taxa de 2023 menos a meta: as duas medem o mesmo passado
HISTORICO = ["mun_taxa_ano_anterior", "mun_distancia_meta_anterior"]

# Mesmas convenções do notebook 05. No SHAP, poucos alunos por município bastam: as features são
# idênticas dentro da cidade, e mais linhas só repetiriam o mesmo vetor.
TETO_TREINO = 30
TETO_SHAP = 3


def variavel_de_origem(nome: str, categoricas: list) -> str:
    """`sigla_uf_RS` -> `sigla_uf`. Numéricas e binárias mantêm o nome."""
    return next((c for c in categoricas if nome.startswith(c + "_")), nome)


def importancia_por_variavel(valores, nomes, categoricas) -> pd.Series:
    """|SHAP| médio por variável original.

    As colunas one-hot de uma mesma variável são somadas linha a linha antes do valor absoluto:
    a contribuição da UF de um aluno é a soma das contribuições das 27 colunas de UF. Sem isso,
    a UF apareceria fragmentada em 27 importâncias pequenas e pareceria irrelevante.
    """
    origem = [variavel_de_origem(n, categoricas) for n in nomes]
    por_variavel = pd.DataFrame(valores, columns=nomes).T.groupby(origem).sum().T
    return por_variavel.abs().mean().sort_values(ascending=False)


def direcao(valores, X_t, nomes, categoricas) -> pd.Series:
    """Sinal da relação entre o valor de cada variável numérica e o seu SHAP.

    Positivo: quanto maior a variável, maior o risco. A correlação de Spearman usa só a ordem,
    então o log e a padronização do pipeline não alteram o sinal.
    """
    sinais = {}
    for j, nome in enumerate(nomes):
        if variavel_de_origem(nome, categoricas) != nome:
            continue
        x, s = pd.Series(X_t[:, j]), pd.Series(valores[:, j])
        if x.nunique() > 1:
            sinais[nome] = x.corr(s, method="spearman")
    return pd.Series(sinais)


def efeito_por_uf(valores, X_t, nomes) -> pd.Series:
    """SHAP médio da coluna de cada UF entre os alunos daquela UF, em log-odds de risco."""
    efeitos = {}
    for j, nome in enumerate(nomes):
        if nome.startswith("sigla_uf_"):
            dentro = X_t[:, j] == 1
            if dentro.any():
                efeitos[nome.replace("sigla_uf_", "")] = valores[dentro, j].mean()
    return pd.Series(efeitos).sort_values()


def interpretar(dados, params: dict, excluir: list) -> dict:
    """Treina sem o RS, avalia no teste sem o RS e calcula o SHAP de uma amostra do teste."""
    pipe, resumo, X_te, y_te, g_te = treinar_e_avaliar(
        NOME, params, dados, excluir=excluir,
        treino_sem_rs=True, teste_sem_rs=True, teto=TETO_TREINO)

    X_shap, _, _ = limitar_por_municipio(X_te, y_te, g_te, TETO_SHAP)
    valores, X_t, nomes = calcular_shap(pipe, X_shap)
    categoricas = carregar_decisoes(excluir=excluir)["categoricas"]

    return {
        "resumo": resumo,
        "importancia": importancia_por_variavel(valores, nomes, categoricas),
        "direcao": direcao(valores, X_t, nomes, categoricas),
        "uf": efeito_por_uf(valores, X_t, nomes),
        "shap": (valores, X_t, nomes),
    }


def plot_comparacao(com: pd.Series, sem: pd.Series, nome_arquivo: str, n: int = 12):
    """Importância das variáveis com e sem o histórico, lado a lado."""
    ordem = list(dict.fromkeys(list(com.index[:n]) + list(sem.index[:n])))
    ordem = sorted(ordem, key=lambda v: sem.get(v, 0) + com.get(v, 0))
    y = np.arange(len(ordem))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(ordem) + 1.2))
    ax.barh(y + 0.2, [com.get(v, 0) for v in ordem], height=0.4, color="#4C78A8",
            label="com histórico")
    ax.barh(y - 0.2, [sem.get(v, 0) for v in ordem], height=0.4, color=COR,
            label="sem histórico")
    ax.set_yticks(y); ax.set_yticklabels(ordem)
    ax.set_xlabel("|SHAP| médio (log-odds de risco)")
    ax.set_title("O que o modelo usa, com e sem o resultado do ano anterior — sem RS")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="x")
    _salvar(fig, nome_arquivo); plt.show()


def interpretacao(dados=None):
    dados = dados or preparar_dados()
    params = pd.read_json(PASTA_RELATORIOS / f"{NOME}_params.json", typ="series")["params"]

    com = interpretar(dados, params, excluir=[])
    sem = interpretar(dados, params, excluir=HISTORICO)

    resumo = pd.DataFrame({"com_historico": com["resumo"], "sem_historico": sem["resumo"]}).T
    ranking = pd.DataFrame({
        "importancia_com_historico": com["importancia"],
        "importancia_sem_historico": sem["importancia"],
        "direcao_com_historico": com["direcao"],
        "direcao_sem_historico": sem["direcao"],
    }).sort_values("importancia_sem_historico", ascending=False)
    uf = pd.DataFrame({"com_historico": com["uf"], "sem_historico": sem["uf"]}) \
        .sort_values("sem_historico")

    resumo.to_csv(PASTA_RELATORIOS / "interpretacao_resumo.csv", encoding="utf-8-sig")
    ranking.to_csv(PASTA_RELATORIOS / "interpretacao_ranking.csv", encoding="utf-8-sig")
    uf.to_csv(PASTA_RELATORIOS / "interpretacao_uf.csv", encoding="utf-8-sig")

    plot_comparacao(com["importancia"], sem["importancia"], "interpretacao_comparacao.png")
    plot_shap(*sem["shap"], "hgb_sem_historico_shap", titulo="HistGradientBoosting sem histórico")
    return resumo, ranking, uf


if __name__ == "__main__":
    resumo, ranking, uf = interpretacao()
    print("=== Desempenho no teste, sem RS ===")
    print(resumo.round(4).to_string())
    print("\n=== Importância por variável — |SHAP| médio, log-odds de risco ===")
    print(ranking.round(3).to_string())
    print("\n=== Efeito médio de cada UF — log-odds de risco ===")
    print(uf.round(3).to_string())
