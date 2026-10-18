"""Árvore de decisão rasa: grade de busca e pipeline com pré-processamento sem escala."""
from sklearn.pipeline import Pipeline

from src.modeling.modelos import criar_modelo
from src.preprocessing.sem_escala import criar_preprocessador_sem_escala

# Árvores baratas permitem busca exaustiva (GridSearchCV, Otimização Aula 2).
# min_samples_leaf como FRAÇÃO das linhas: a busca roda numa amostra (~126 mil linhas) e o treino
# final usa a base inteira (~1,3 milhão); a fração mantém o mesmo tamanho relativo de folha.
GRADE_BUSCA = {
    "modelo__max_depth": [2, 3, 4, 5, 6, 8],
    "modelo__min_samples_leaf": [0.001, 0.002, 0.005, 0.01],
}


def criar_pipeline_arvore(decisoes: dict, **params) -> Pipeline:
    """Imputação + one-hot (sem escala) + DecisionTreeClassifier (class_weight='balanced')."""
    return Pipeline([("pre", criar_preprocessador_sem_escala(decisoes)),
                     ("modelo", criar_modelo("arvore_rasa", **params))])
