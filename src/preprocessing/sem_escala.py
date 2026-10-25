"""Pré-processamento sem log nem padronização, para modelos de árvore.

Árvores são invariantes a transformações monotônicas (log, padronização): as divisões que
escolhem são as mesmas. Sem a escala, os limiares das regras ficam em unidades originais
(ex.: "mun_taxa_ano_anterior <= 0.62" em vez de um z-score), que é o que um gestor consegue ler.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing.pipeline import LimitarExtremos, separar_limitadas


def criar_preprocessador_sem_escala(decisoes: dict) -> ColumnTransformer:
    """Mesma imputação, mesmo teto de extremos e mesmo one-hot do pré-processador padrão,
    sem log e sem padronização.

    O teto não é escala: corrige valores impossíveis, então se aplica aqui também. Para a
    árvore o efeito é pequeno — os limiares das regras continuam em unidades originais.
    """
    imp = decisoes["imputacao"]
    limitadas, padrao, percentil = separar_limitadas(decisoes, "numericas_padrao")
    numericas = SimpleImputer(strategy=imp["numericas"])
    numericas_limitadas = Pipeline([
        ("imputar", SimpleImputer(strategy=imp["numericas"])),
        ("limitar", LimitarExtremos(percentil_superior=percentil)),
    ])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy=imp["categoricas"])),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binarias = SimpleImputer(strategy="constant", fill_value=imp["binarias_valor"])
    return ColumnTransformer([
        ("num", numericas, decisoes["numericas_log"] + padrao),
        ("num_lim", numericas_limitadas, limitadas),
        ("cat", categoricas, decisoes["categoricas"]),
        ("bin", binarias, decisoes["binarias"]),
    ])
