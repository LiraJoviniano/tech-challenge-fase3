"""Por que os modelos não usam class_weight — o experimento que levou à decisão.

Os quatro modelos foram construídos inicialmente com class_weight="balanced", para priorizar o
recall da classe de risco. O peso faz o modelo aprender como se metade dos alunos estivesse em
risco, quando a taxa real é de cerca de 41%.

Este módulo treina cada modelo com e sem o peso, na mesma amostra, e compara no mesmo teste:

- **Ordenação** — ROC-AUC e PR-AUC não dependem de limiar. Se ficam iguais, o peso não ajuda a
  distinguir quem está em risco, e o recall-alvo pode ser obtido só pela escolha do limiar.
- **Probabilidades** — o Brier, o erro de calibração e o risco médio previsto mostram se o
  número que o modelo diz corresponde ao que acontece.

O resultado levou a retirar o peso: a ordenação é a mesma, e as probabilidades sem ele já saem
calibradas. Com ele, seria preciso corrigi-las depois — conserto de um problema que o desenho
mais simples não cria.

Uso: python -m src.evaluation.experimento_class_weight
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.avaliacao import PASTA_RELATORIOS
from src.evaluation.calibracao import erro_esperado_calibracao, faixas_de_calibracao
from src.modeling.arvore import criar_pipeline_arvore
from src.modeling.modelos import NOMES_LEGIVEIS, criar_pipeline, limitar_por_municipio
from src.preprocessing.pipeline import listar_features, preparar_dados
from src.visualization.graficos import COR, _salvar

MODELOS = ["regressao_logistica", "arvore_rasa", "random_forest", "hist_gradient_boosting"]

# Mesma amostragem do treino dos notebooks: até 30 alunos por município
TETO = 30


def montar(nome: str, decisoes: dict, peso):
    """O mesmo pipeline do projeto, com ou sem o peso balanceado."""
    if nome == "arvore_rasa":
        return criar_pipeline_arvore(decisoes, class_weight=peso)
    return criar_pipeline(nome, decisoes, class_weight=peso)


def avaliar(y, proba) -> dict:
    tabela = faixas_de_calibracao(y, proba)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "ece_pontos": 100 * erro_esperado_calibracao(tabela),
        "risco_medio_previsto_pct": 100 * float(np.mean(proba)),
        "risco_observado_pct": 100 * float(np.mean(y)),
    }


def plot_experimento(curvas: dict, nome_arquivo: str):
    """Diagrama de confiabilidade do modelo escolhido, com e sem o peso."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="calibração perfeita")
    for (rotulo, tabela), cor in zip(curvas.items(), ("#4C78A8", COR)):
        ax.plot(tabela["risco_previsto"], tabela["risco_observado"], "o-", color=cor,
                linewidth=2, markersize=5, label=rotulo)
    ax.set_xlabel("Risco previsto pelo modelo"); ax.set_ylabel("Risco observado no teste")
    ax.set_title("HistGradientBoosting com e sem class_weight — teste, sem RS")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    _salvar(fig, nome_arquivo); plt.show()


def experimento(dados=None) -> pd.DataFrame:
    dados = dados or preparar_dados()
    cols = listar_features(dados.decisoes)
    X, y, _ = limitar_por_municipio(dados.X_treino, dados.y_treino, dados.g_treino, TETO)

    sem_rs = (dados.X_teste["sigla_uf"] != "RS").values
    y_teste = dados.y_teste.values[sem_rs]

    linhas, curvas = [], {}
    for nome in MODELOS:
        for rotulo, peso in (("com class_weight", "balanced"), ("sem class_weight", None)):
            pipe = montar(nome, dados.decisoes, peso).fit(X[cols], y)
            proba = pipe.predict_proba(dados.X_teste[cols])[:, 1][sem_rs]
            linhas.append({"modelo": NOMES_LEGIVEIS[nome], "versao": rotulo,
                           **avaliar(y_teste, proba)})
            if nome == "hist_gradient_boosting":
                curvas[rotulo] = faixas_de_calibracao(y_teste, proba)

    resultado = pd.DataFrame(linhas)
    resultado.to_csv(PASTA_RELATORIOS / "experimento_class_weight.csv", index=False,
                     encoding="utf-8-sig")
    plot_experimento(curvas, "experimento_class_weight.png")
    return resultado


if __name__ == "__main__":
    tabela = experimento()
    print(tabela.round(4).to_string(index=False))
