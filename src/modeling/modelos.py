"""Catálogo dos modelos, espaços de busca e amostragem por município.

Todos usam class_weight="balanced" (a métrica prioritária é o recall da classe de risco).
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.pipeline import RANDOM_STATE, criar_preprocessador

# Parâmetros-base de cada modelo; `criar_modelo(nome, **params)` sobrescreve o que for passado.
_BASE = {
    "regressao_logistica": (LogisticRegression, dict(max_iter=1000, class_weight="balanced")),
    "arvore_rasa": (DecisionTreeClassifier, dict(
        max_depth=5, min_samples_leaf=200, class_weight="balanced", random_state=RANDOM_STATE)),
    "random_forest": (RandomForestClassifier, dict(
        n_estimators=100, max_depth=12, min_samples_leaf=50, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE)),
    # early_stopping=False de propósito: a validação interna é aleatória por aluno e misturaria
    # alunos do mesmo município no treino e na validação (parada otimista demais).
    "hist_gradient_boosting": (HistGradientBoostingClassifier, dict(
        max_depth=4, learning_rate=0.05, max_iter=200, class_weight="balanced",
        early_stopping=False, random_state=RANDOM_STATE)),
}

NOMES_LEGIVEIS = {
    "regressao_logistica": "Regressão logística",
    "arvore_rasa": "Árvore de decisão (rasa)",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}

# Espaços para o RandomizedSearchCV (prefixo "modelo__" = etapa do Pipeline).
ESPACOS_BUSCA = {
    "random_forest": {
        "modelo__n_estimators": [100, 200],
        "modelo__max_depth": [6, 8, 12, 16],
        "modelo__min_samples_leaf": [20, 50, 100, 200],
        "modelo__max_features": ["sqrt", 0.3],
    },
    "hist_gradient_boosting": {
        "modelo__learning_rate": [0.03, 0.05, 0.1],
        "modelo__max_depth": [3, 4, 6],
        "modelo__max_leaf_nodes": [15, 31],
        "modelo__min_samples_leaf": [50, 100, 200],
        "modelo__l2_regularization": [0.0, 1.0, 10.0],
        "modelo__max_iter": [100, 200, 300],
    },
}


def criar_modelo(nome: str, **params):
    classe, base = _BASE[nome]
    return classe(**{**base, **params})


def criar_pipeline(nome: str, decisoes: dict, **params) -> Pipeline:
    """Pré-processamento + modelo num único Pipeline (o pré-processamento é reajustado em cada fold)."""
    return Pipeline([("pre", criar_preprocessador(decisoes)),
                     ("modelo", criar_modelo(nome, **params))])


def limitar_por_municipio(X, y, grupos, teto: int):
    """Até `teto` alunos por município, sorteados com semente fixa.

    As features são municipais: 1,3 milhão de alunos equivale a ~4.400 pontos independentes.
    O teto acelera a busca/interpretação sem perder municípios e evita que as capitais dominem.
    Para treino final e métricas do teste, usar a base completa.
    """
    if teto is None:   # sem teto: base inteira
        return X, y, grupos
    idx = (X.assign(_g=grupos.values).groupby("_g", group_keys=False)
             .apply(lambda s: s.sample(min(len(s), teto), random_state=RANDOM_STATE)).index)
    return X.loc[idx], y.loc[idx], grupos.loc[idx]
