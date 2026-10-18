"""Métricas, escolha de limiar, interpretação e gravação dos resultados de cada modelo."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

from src.modeling.modelos import criar_modelo, criar_pipeline, limitar_por_municipio
from src.preprocessing.pipeline import (RANDOM_STATE, RAIZ, carregar_decisoes, criar_preprocessador,
                                        listar_features, sem_rs)

PASTA_RELATORIOS = RAIZ / "reports" / "modelagem"
PASTA_IMAGENS = RAIZ / "images"


def metricas(y_real, proba, limiar: float) -> dict:
    """Métricas no limiar dado. ROC-AUC e PR-AUC não dependem do limiar."""
    previsto = (proba >= limiar).astype(int)
    return {
        "roc_auc": roc_auc_score(y_real, proba),
        "pr_auc": average_precision_score(y_real, proba),
        "recall": recall_score(y_real, previsto),
        "precisao": precision_score(y_real, previsto),
        "f1": f1_score(y_real, previsto),
        "alertas (% dos alunos)": previsto.mean(),
    }


def probas_fora_da_amostra(pipeline, X, y, grupos, n_splits: int = 5):
    """Probabilidades de risco previstas por validação cruzada agrupada por município."""
    return cross_val_predict(pipeline, X, y, groups=grupos, cv=GroupKFold(n_splits=n_splits),
                             method="predict_proba")[:, 1]


def escolher_limiar(y, proba, meta_recall: float) -> dict:
    """Maior limiar que ainda atinge `meta_recall` (assim a precisão é a maior possível)."""
    precisao, recall, limiares = precision_recall_curve(y, proba)
    atingem = np.where(recall[:-1] >= meta_recall)[0]
    i = atingem[-1]
    return {"limiar": float(limiares[i]), "recall": float(recall[i]), "precisao": float(precisao[i]),
            "curva_precisao": precisao, "curva_recall": recall}


def treinar_e_avaliar(nome_modelo, params, dados, excluir=(), treino_sem_rs=False,
                      teste_sem_rs=False, teto=30):
    """Treina numa amostra (teto por município) e avalia no teste. Usado nas análises de sensibilidade.

    Retorna (pipeline, métricas de ranking, X_teste, y_teste, g_teste), com X_teste já restrito
    às features usadas (pronto para permutation importance).
    """
    decisoes = carregar_decisoes(excluir=list(excluir))
    cols = listar_features(decisoes)
    X_tr, y_tr, g_tr = dados.X_treino, dados.y_treino, dados.g_treino
    X_te, y_te, g_te = dados.X_teste, dados.y_teste, dados.g_teste
    if treino_sem_rs:
        X_tr, y_tr, g_tr = sem_rs(X_tr, y_tr, g_tr)
    if teste_sem_rs:
        X_te, y_te, g_te = sem_rs(X_te, y_te, g_te)
    X_am, y_am, _ = limitar_por_municipio(X_tr, y_tr, g_tr, teto)
    pipe = criar_pipeline(nome_modelo, decisoes, **params).fit(X_am[cols], y_am)
    proba = pipe.predict_proba(X_te[cols])[:, 1]
    resumo = {"roc_auc": roc_auc_score(y_te, proba), "pr_auc": average_precision_score(y_te, proba)}
    return pipe, resumo, X_te[cols], y_te, g_te


def importancia_permutacao(pipeline, X, y, n_repeats: int = 5) -> pd.Series:
    """Queda do ROC-AUC ao embaralhar cada feature original (o pipeline cuida do pré-processamento)."""
    perm = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    return pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)


def calcular_shap(pipeline, X_bruto):
    """Valores SHAP da classe de risco. Trata os dois formatos de saída do shap (lista ou array 3D)."""
    import shap
    pre, modelo = pipeline.named_steps["pre"], pipeline.named_steps["modelo"]
    X_t = pre.transform(X_bruto)
    nomes = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    valores = shap.TreeExplainer(modelo).shap_values(X_t, check_additivity=False)
    if isinstance(valores, list):
        valores = valores[1]
    elif valores.ndim == 3:
        valores = valores[:, :, 1]
    return valores, X_t, nomes


def curva_por_iteracao(decisoes, X, y, grupos, params, max_iter: int = 600, passo: int = 10):
    """AUC de treino e de validação a cada `passo` árvores do boosting (validação separada por município)."""
    i_tr, i_va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                      .split(X, y, groups=grupos))
    pre = criar_preprocessador(decisoes).fit(X.iloc[i_tr])
    X_tr, X_va = pre.transform(X.iloc[i_tr]), pre.transform(X.iloc[i_va])
    y_tr, y_va = y.iloc[i_tr], y.iloc[i_va]
    modelo = criar_modelo("hist_gradient_boosting", **{**params, "max_iter": max_iter}).fit(X_tr, y_tr)
    iteracoes, auc_tr, auc_va = [], [], []
    for k, (p_tr, p_va) in enumerate(zip(modelo.staged_predict_proba(X_tr),
                                         modelo.staged_predict_proba(X_va)), start=1):
        if k % passo == 0:
            iteracoes.append(k)
            auc_tr.append(roc_auc_score(y_tr, p_tr[:, 1]))
            auc_va.append(roc_auc_score(y_va, p_va[:, 1]))
    return np.array(iteracoes), np.array(auc_tr), np.array(auc_va)


def salvar_resultados(nome, pipeline, dados, limiar, meta_recall, params, rotulo_modelo=None):
    """Avalia no teste completo e grava tabela, parâmetros/limiar e probabilidades (para comparar modelos)."""
    PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
    proba = pipeline.predict_proba(dados.X_teste)[:, 1]
    tabela = pd.DataFrame({
        "limiar 0.50": metricas(dados.y_teste, proba, 0.5),
        f"limiar {limiar:.2f} (escolhido)": metricas(dados.y_teste, proba, limiar),
    }).T
    tabela.to_csv(PASTA_RELATORIOS / f"{nome}_teste.csv", encoding="utf-8-sig")
    (PASTA_RELATORIOS / f"{nome}_params.json").write_text(json.dumps(
        {"modelo": rotulo_modelo or nome, "params": params, "limiar": limiar, "meta_recall": meta_recall},
        indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"proba": proba}, index=dados.y_teste.index).to_parquet(
        PASTA_RELATORIOS / f"probas_teste_{nome}.parquet")
    return proba, tabela
