"""Treina um modelo com parâmetros-padrão do catálogo (sem busca), escolhe o limiar e grava os resultados.

Serve para o baseline (regressão logística) e para a árvore rasa. Uso:
    python -m src.modeling.treinar --modelo regressao_logistica
    python -m src.modeling.treinar --modelo arvore_rasa
"""
import argparse
import time
import warnings

from src.evaluation.avaliacao import (escolher_limiar, probas_fora_da_amostra, salvar_resultados)
from src.modeling.modelos import NOMES_LEGIVEIS, _BASE, criar_pipeline, limitar_por_municipio
from src.preprocessing.pipeline import preparar_dados


def treinar_sem_busca(nome, dados, meta_recall=0.70, teto=30, params=None):
    params = params or {}
    X_am, y_am, g_am = limitar_por_municipio(dados.X_treino, dados.y_treino, dados.g_treino, teto)
    pipe = criar_pipeline(nome, dados.decisoes, **params)

    proba_oof = probas_fora_da_amostra(pipe, X_am, y_am, g_am)
    escolha = escolher_limiar(y_am, proba_oof, meta_recall)

    modelo_final = criar_pipeline(nome, dados.decisoes, **params).fit(dados.X_treino, dados.y_treino)
    proba, tabela = salvar_resultados(nome, modelo_final, dados, escolha["limiar"], meta_recall,
                                      params or "padrão do catálogo", NOMES_LEGIVEIS[nome])
    return {"pipeline": modelo_final, "limiar": escolha["limiar"], "tabela": tabela, "proba": proba}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=list(_BASE))
    ap.add_argument("--teto", type=int, default=30)
    ap.add_argument("--meta-recall", type=float, default=0.70)
    a = ap.parse_args()
    t0 = time.time()
    res = treinar_sem_busca(a.modelo, preparar_dados(), a.meta_recall, a.teto)
    print(res["tabela"].round(3).to_string())
    print(f"limiar {res['limiar']:.3f} | {time.time() - t0:.0f}s")
