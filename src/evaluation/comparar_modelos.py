"""Compara os modelos do catálogo por validação cruzada agrupada por município (só no treino).

Uso:  python -m src.evaluation.comparar_modelos --teto 30
"""
import argparse
import time
import warnings

import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate

from src.modeling.modelos import NOMES_LEGIVEIS, criar_pipeline, limitar_por_municipio
from src.evaluation.avaliacao import PASTA_RELATORIOS
from src.preprocessing.pipeline import preparar_dados

METRICAS = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall",
            "precisao": "precision", "f1": "f1", "acuracia": "accuracy"}


def main(teto):
    warnings.filterwarnings("ignore")
    dados = preparar_dados()
    X, y, g = limitar_por_municipio(dados.X_treino, dados.y_treino, dados.g_treino, teto)
    print(f"Treino: {len(y):,} alunos, {g.nunique():,} municípios, risco {y.mean():.1%}")
    linhas = {}
    for nome, rotulo in NOMES_LEGIVEIS.items():
        t0 = time.time()
        cv = cross_validate(criar_pipeline(nome, dados.decisoes), X, y, groups=g,
                            cv=GroupKFold(n_splits=5), scoring=METRICAS)
        linhas[rotulo] = {m: f"{cv[f'test_{m}'].mean():.3f} ± {cv[f'test_{m}'].std():.3f}" for m in METRICAS}
        print(f"{rotulo}: ok ({time.time() - t0:.0f}s)")
    tabela = pd.DataFrame(linhas).T
    print(tabela.to_string())
    PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(PASTA_RELATORIOS / f"comparacao_modelos_cv_teto{teto}.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teto", type=int, default=30)
    main(ap.parse_args().teto)
