"""Gráficos da modelagem. Cada função grava a figura em images/ e a exibe."""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

from src.evaluation.avaliacao import PASTA_IMAGENS

COR = "#D30F59"


def _salvar(fig, nome):
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_IMAGENS / nome, dpi=150, bbox_inches="tight")


def plot_precisao_recall_cv(escolha, taxa_base, nome_arquivo):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(escolha["curva_recall"], escolha["curva_precisao"], color=COR, linewidth=2)
    ax.scatter([escolha["recall"]], [escolha["precisao"]], color="black", zorder=3,
               label=f"limiar {escolha['limiar']:.2f}")
    ax.axhline(taxa_base, color="gray", linestyle="--", label="acaso (taxa de risco)")
    ax.set_xlabel("Recall (classe de risco)"); ax.set_ylabel("Precisão")
    ax.set_title("Curva precisão-recall (validação cruzada por município)")
    ax.legend(); ax.grid(alpha=0.3)
    _salvar(fig, nome_arquivo); plt.show()


def plot_avaliacao_teste(y, proba, limiar, nome_modelo, nome_arquivo):
    previsto = (proba >= limiar).astype(int)
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(confusion_matrix(y, previsto),
                           display_labels=["Alfabetizado", "Não alfabetizado"]).plot(
        ax=eixos[0], cmap="Blues", values_format="d", colorbar=False)
    eixos[0].set_title(f"Matriz de confusão no teste (limiar {limiar:.2f})")
    fpr, tpr, _ = roc_curve(y, proba)
    eixos[1].plot(fpr, tpr, color=COR, linewidth=2, label=f"{nome_modelo} (AUC {roc_auc_score(y, proba):.3f})")
    eixos[1].plot([0, 1], [0, 1], "--", color="gray", label="acaso")
    eixos[1].set_xlabel("Taxa de falsos positivos"); eixos[1].set_ylabel("Recall")
    eixos[1].set_title("Curva ROC no teste"); eixos[1].legend(); eixos[1].grid(alpha=0.3)
    _salvar(fig, nome_arquivo); plt.show()


def plot_importancias(imp_perm, imp_modelo, titulo_modelo, nome_arquivo, n=12):
    fig, eixos = plt.subplots(1, 2, figsize=(14, 6))
    imp_modelo.head(n)[::-1].plot.barh(ax=eixos[0], color="#4C78A8")
    eixos[0].set_title(f"Importância interna do modelo (top {n}, após o pré-processamento)")
    imp_perm.head(n)[::-1].plot.barh(ax=eixos[1], color=COR)
    eixos[1].set_title("Permutation importance no teste (queda do ROC-AUC)")
    fig.suptitle(titulo_modelo)
    plt.tight_layout(); _salvar(fig, nome_arquivo); plt.show()


def plot_shap(valores, X_t, nomes, prefixo_arquivo, titulo=""):
    import shap
    for tipo, sufixo, rotulo in [(None, "summary", "impacto de cada feature"),
                                 ("bar", "bar", "importância média absoluta")]:
        plt.figure()
        shap.summary_plot(valores, X_t, feature_names=nomes, plot_type=tipo, show=False, max_display=15)
        plt.title(f"SHAP {titulo}: {rotulo}")
        _salvar(plt.gcf(), f"{prefixo_arquivo}_{sufixo}.png"); plt.show()


def plot_curva_iteracoes(iteracoes, auc_treino, auc_validacao, nome_arquivo):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iteracoes, auc_treino, color="#4C78A8", label="treino")
    ax.plot(iteracoes, auc_validacao, color=COR, label="validação (municípios inéditos)")
    melhor = iteracoes[auc_validacao.argmax()]
    ax.axvline(melhor, color="gray", linestyle="--", label=f"melhor validação: {melhor} árvores")
    ax.set_xlabel("Número de árvores (iterações do boosting)"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Overfitting no boosting: AUC por iteração"); ax.legend(); ax.grid(alpha=0.3)
    _salvar(fig, nome_arquivo); plt.show()
