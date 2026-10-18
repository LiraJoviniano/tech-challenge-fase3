"""Carga dos dados, divisão por município e pré-processamento da modelagem supervisionada.

As decisões de quais features usar e como tratá-las ficam em src/config/features_modelo.json
(resultado do notebook 02). Nada aqui olha o conjunto de teste.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_GOLD = RAIZ / "data" / "sample" / "features_aluno"
CAMINHO_DECISOES = RAIZ / "src" / "config" / "features_modelo.json"
RANDOM_STATE = 42

GRUPOS_DE_FEATURES = ("numericas_log", "numericas_padrao", "categoricas", "binarias")


def carregar_decisoes(excluir=None) -> dict:
    """Lê as decisões de features. `excluir` remove colunas (útil para análises de sensibilidade)."""
    decisoes = json.loads(CAMINHO_DECISOES.read_text(encoding="utf-8"))
    if excluir:
        for grupo in GRUPOS_DE_FEATURES:
            decisoes[grupo] = [c for c in decisoes[grupo] if c not in set(excluir)]
    return decisoes


def listar_features(decisoes: dict) -> list:
    return [c for grupo in GRUPOS_DE_FEATURES for c in decisoes[grupo]]


@dataclass
class Dados:
    """Treino e teste já separados por município (nenhum município aparece nos dois)."""
    X_treino: pd.DataFrame
    X_teste: pd.DataFrame
    y_treino: pd.Series
    y_teste: pd.Series
    g_treino: pd.Series
    g_teste: pd.Series
    decisoes: dict


def preparar_dados(test_size: float = 0.2) -> Dados:
    """Monta X, y (1 = NÃO alfabetizado, a classe de risco) e divide por município.

    X sempre traz todas as features do JSON; o ColumnTransformer escolhe as que cada
    modelo usa. A divisão depende só do município, então é idêntica em qualquer cenário.
    """
    decisoes = carregar_decisoes()
    df = pd.read_parquet(CAMINHO_GOLD)
    X = df[listar_features(decisoes)].copy()
    for coluna in decisoes["binarias"]:
        X[coluna] = X[coluna].astype(float)  # True/False/vazio -> 1.0/0.0/NaN
    y = (~df[decisoes["alvo"]]).astype(int).rename("nao_alfabetizado")
    grupos = df[decisoes["coluna_grupo"]]

    divisor = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    i_tr, i_te = next(divisor.split(X, y, groups=grupos))
    dados = Dados(X.iloc[i_tr], X.iloc[i_te], y.iloc[i_tr], y.iloc[i_te],
                  grupos.iloc[i_tr], grupos.iloc[i_te], decisoes)
    assert set(dados.g_treino).isdisjoint(set(dados.g_teste)), "município em treino e teste"
    return dados


def sem_rs(X, y, grupos):
    """Remove os alunos do RS (anomalia de origem documentada na Fase 2)."""
    mascara = (X["sigla_uf"] != "RS").values
    return X[mascara], y[mascara], grupos[mascara]


def criar_preprocessador(decisoes: dict) -> ColumnTransformer:
    """Imputação + transformação por grupo de feature. Sempre dentro de um Pipeline."""
    imp = decisoes["imputacao"]
    numericas_log = Pipeline([
        ("imputar", SimpleImputer(strategy=imp["numericas"])),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("padronizar", StandardScaler()),
    ])
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy=imp["numericas"])),
        ("padronizar", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy=imp["categoricas"])),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binarias = SimpleImputer(strategy="constant", fill_value=imp["binarias_valor"])
    return ColumnTransformer([
        ("num_log", numericas_log, decisoes["numericas_log"]),
        ("num", numericas, decisoes["numericas_padrao"]),
        ("cat", categoricas, decisoes["categoricas"]),
        ("bin", binarias, decisoes["binarias"]),
    ])
