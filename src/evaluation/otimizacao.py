"""Otimização do modelo escolhido — duas verificações que a busca sozinha não faz.

1. **A configuração escolhida está no ponto certo de iterações?** O notebook 05 mostra o
   overfitting com uma configuração agressiva, de propósito. Esta verificação usa a configuração
   que a busca de fato escolheu e localiza o pico da validação.

2. **Por que o teste difere da validação?** A busca e o limiar são escolhidos numa amostra de até
   30 alunos por município, em que cada cidade pesa quase igual. O teste é reportado por aluno,
   e os municípios grandes pesam mais. Medir o teste dos dois jeitos separa o que é perda de
   generalização do que é só forma de contar.

Uso: python -m src.evaluation.otimizacao
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, roc_auc_score

from src.evaluation.avaliacao import PASTA_RELATORIOS, curva_por_iteracao
from src.modeling.modelos import limitar_por_municipio
from src.preprocessing.pipeline import preparar_dados
from src.visualization.graficos import COR, _salvar

NOME = "hist_gradient_boosting"
TETO = 30
MAX_ITER = 600
PASSO = 10


def curva_final(dados, params: dict) -> pd.DataFrame:
    """ROC-AUC de treino e validação por número de árvores, na configuração escolhida."""
    sem_iter = {k: v for k, v in params.items() if k != "max_iter"}
    X, y, g = limitar_por_municipio(dados.X_treino, dados.y_treino, dados.g_treino, TETO)
    iteracoes, treino, validacao = curva_por_iteracao(dados.decisoes, X, y, g, sem_iter,
                                                      max_iter=MAX_ITER, passo=PASSO)
    return pd.DataFrame({"arvores": iteracoes, "treino": treino, "validacao": validacao})


def plot_curva(curva: pd.DataFrame, escolhidas: int, nome_arquivo: str):
    pico = curva.loc[curva["validacao"].idxmax()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curva["arvores"], curva["treino"], color="#4C78A8", label="treino")
    ax.plot(curva["arvores"], curva["validacao"], color=COR, label="validação (municípios inéditos)")
    ax.axvline(pico["arvores"], color="gray", linestyle=":",
               label=f"pico da validação: {int(pico['arvores'])} árvores")
    ax.axvline(escolhidas, color="gray", linestyle="--", label=f"configuração escolhida: {escolhidas}")
    ax.set_xlabel("Número de árvores"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Configuração escolhida: ROC-AUC por iteração")
    ax.legend(); ax.grid(alpha=0.3)
    _salvar(fig, nome_arquivo); plt.show()


def validacao_versus_teste(dados, limiar: float, auc_busca: float) -> pd.DataFrame:
    """O teste medido do jeito que a busca mede — até 30 por município — e por aluno."""
    proba = pd.read_parquet(PASTA_RELATORIOS / f"probas_teste_{NOME}.parquet")["proba"]
    _, y30, _ = limitar_por_municipio(dados.X_teste, dados.y_teste, dados.g_teste, TETO)
    p30 = proba.loc[y30.index]

    _, y_am, _ = limitar_por_municipio(dados.X_treino, dados.y_treino, dados.g_treino, TETO)
    return pd.DataFrame([
        {"medida": "validação cruzada da busca", "contagem": "até 30 por município",
         "roc_auc": auc_busca, "recall_no_limiar": np.nan, "risco": y_am.mean()},
        {"medida": "teste", "contagem": "até 30 por município",
         "roc_auc": roc_auc_score(y30, p30), "recall_no_limiar": recall_score(y30, p30 >= limiar),
         "risco": y30.mean()},
        {"medida": "teste", "contagem": "todos os alunos",
         "roc_auc": roc_auc_score(dados.y_teste, proba),
         "recall_no_limiar": recall_score(dados.y_teste, proba >= limiar),
         "risco": dados.y_teste.mean()},
    ])


def otimizacao(dados=None):
    dados = dados or preparar_dados()
    params = pd.read_json(PASTA_RELATORIOS / f"{NOME}_params.json", typ="series")["params"]
    limiar = pd.read_csv(PASTA_RELATORIOS / "comparacao_teste.csv", index_col=0) \
        .loc["HistGradientBoosting", "limiar"]
    auc_busca = pd.read_csv(PASTA_RELATORIOS / f"{NOME}_busca.csv")["roc_auc"].max()

    curva = curva_final(dados, params)
    comparacao = validacao_versus_teste(dados, limiar, auc_busca)

    curva.to_csv(PASTA_RELATORIOS / "otimizacao_curva_final.csv", index=False, encoding="utf-8-sig")
    comparacao.to_csv(PASTA_RELATORIOS / "otimizacao_validacao_teste.csv", index=False,
                      encoding="utf-8-sig")
    plot_curva(curva, int(params["max_iter"]), "hgb_curva_iteracoes_final.png")
    return params, curva, comparacao


if __name__ == "__main__":
    params, curva, comparacao = otimizacao()
    pico = curva.loc[curva["validacao"].idxmax()]
    escolhida = curva.loc[curva["arvores"] == params["max_iter"]].iloc[0]
    print(f"Configuração escolhida: {params}")
    print(f"Pico da validação: {pico['validacao']:.4f} com {int(pico['arvores'])} árvores")
    print(f"Escolhida ({int(params['max_iter'])} árvores): validação {escolhida['validacao']:.4f}, "
          f"treino {escolhida['treino']:.4f}")
    print("\n=== Validação versus teste ===")
    print(comparacao.round(4).to_string(index=False))
