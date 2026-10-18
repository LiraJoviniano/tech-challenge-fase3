"""Gráficos específicos da árvore de decisão (curva de validação e desenho da árvore)."""
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from src.visualization.graficos import COR, _salvar


def plot_curva_validacao(valores, treino, validacao, desvio, rotulo_eixo, nome_arquivo):
    """AUC de treino e de validação em função de um hiperparâmetro (Otimização, Aula 5, slide16)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(valores, treino, color="#4C78A8", marker="o", label="treino")
    ax.plot(valores, validacao, color=COR, marker="o", label="validação (municípios inéditos)")
    ax.fill_between(valores, validacao - desvio, validacao + desvio, color=COR, alpha=0.15)
    ax.set_xlabel(rotulo_eixo)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Curva de validação: complexidade da árvore")
    ax.legend()
    ax.grid(alpha=0.3)
    _salvar(fig, nome_arquivo)
    plt.show()


def plot_arvore(pipeline, nome_arquivo, profundidade_max=3):
    """Desenha os primeiros níveis da árvore (nomes das features em unidades originais)."""
    pre, arvore = pipeline.named_steps["pre"], pipeline.named_steps["modelo"]
    nomes = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(arvore, feature_names=nomes, class_names=["alfabetizado", "não alfabetizado"],
              filled=True, rounded=True, max_depth=profundidade_max, proportion=True,
              fontsize=9, ax=ax)
    ax.set_title(f"Árvore de decisão: primeiros {profundidade_max} níveis "
                 "(cor = classe majoritária ponderada; porcentagens = fração dos alunos)")
    _salvar(fig, nome_arquivo)
    plt.show()
