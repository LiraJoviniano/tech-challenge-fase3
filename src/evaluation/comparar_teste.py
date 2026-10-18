"""Compara, no MESMO conjunto de teste, todos os modelos que já gravaram `probas_teste_<modelo>.parquet`.

Cada modelo usa o seu próprio limiar (escolhido na validação cruzada para o mesmo recall-alvo),
então recall e precisão ficam comparáveis. Uso: python -m src.evaluation.comparar_teste
"""
import json

import pandas as pd

from src.evaluation.avaliacao import PASTA_RELATORIOS, metricas
from src.preprocessing.pipeline import preparar_dados


def comparar_teste(dados=None) -> pd.DataFrame:
    dados = dados or preparar_dados()
    linhas = {}
    for arq in sorted(PASTA_RELATORIOS.glob("probas_teste_*.parquet")):
        nome = arq.stem.replace("probas_teste_", "")
        info = json.loads((PASTA_RELATORIOS / f"{nome}_params.json").read_text(encoding="utf-8"))
        proba = pd.read_parquet(arq)["proba"]
        assert proba.index.equals(dados.y_teste.index), f"{nome}: teste diferente do atual"
        linha = metricas(dados.y_teste, proba.values, info["limiar"])
        linha["limiar"] = info["limiar"]
        linhas[info["modelo"]] = linha
    tabela = pd.DataFrame(linhas).T.sort_values("roc_auc", ascending=False)
    tabela.to_csv(PASTA_RELATORIOS / "comparacao_teste.csv", encoding="utf-8-sig")
    return tabela


if __name__ == "__main__":
    print(comparar_teste().round(3).to_string())
