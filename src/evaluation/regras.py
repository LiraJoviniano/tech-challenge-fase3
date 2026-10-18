"""Transforma uma árvore de decisão ajustada em uma tabela de regras (uma linha por folha)."""
import numpy as np
import pandas as pd


def regras_da_arvore(pipeline, conjuntos: dict, casas: int = 2) -> pd.DataFrame:
    """Uma linha por folha: regra, previsão e, para cada conjunto, alunos, municípios e taxa de risco.

    `pipeline` = Pipeline([("pre", pré-processador sem escala), ("modelo", árvore)]) já ajustado.
    `conjuntos` = {"treino": (X, y, grupos), "teste": (X, y, grupos)}; y = 1 é não alfabetizado.
    A taxa de risco é a fração REAL de alunos não alfabetizados naquela folha (não a probabilidade
    ponderada pelo class_weight), então dá para comparar treino e teste.
    """
    pre, arvore = pipeline.named_steps["pre"], pipeline.named_steps["modelo"]
    brutos = list(pre.get_feature_names_out())
    nomes = [n.split("__", 1)[1] for n in brutos]
    booleanas = {n.split("__", 1)[1] for n in brutos if n.startswith(("cat__", "bin__"))}
    t = arvore.tree_

    regras = {}

    def descer(no, condicoes):
        if t.children_left[no] == t.children_right[no]:   # folha
            regras[no] = condicoes
            return
        nome, limite = nomes[t.feature[no]], t.threshold[no]
        if nome in booleanas:   # one-hot ou binária: o limiar é sempre 0,5
            esquerda, direita = f"{nome} = não", f"{nome} = sim"
        else:
            esquerda, direita = f"{nome} <= {limite:.{casas}f}", f"{nome} > {limite:.{casas}f}"
        descer(t.children_left[no], condicoes + [esquerda])
        descer(t.children_right[no], condicoes + [direita])

    descer(0, [])

    linhas = {}
    for no, condicoes in regras.items():
        classe = int(np.argmax(t.value[no][0]))
        linhas[no] = {
            "regra": " E ".join(condicoes) if condicoes else "(todos)",
            "previsao": "não alfabetizado" if classe == 1 else "alfabetizado",
            "profundidade": len(condicoes),
        }

    for rotulo, (X, y, grupos) in conjuntos.items():
        folhas = arvore.apply(pre.transform(X))
        y_arr, g_arr = np.asarray(y), np.asarray(grupos)
        for no in regras:
            mascara = folhas == no
            n = int(mascara.sum())
            linhas[no][f"alunos_{rotulo}"] = n
            linhas[no][f"municipios_{rotulo}"] = int(len(np.unique(g_arr[mascara]))) if n else 0
            linhas[no][f"risco_{rotulo}"] = float(y_arr[mascara].mean()) if n else np.nan

    tabela = pd.DataFrame(list(linhas.values()))
    primeiro = next(iter(conjuntos))
    return tabela.sort_values(f"risco_{primeiro}", ascending=False).reset_index(drop=True)
