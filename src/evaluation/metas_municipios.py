"""Quais municípios podem não atingir a meta — pergunta de negócio do enunciado.

Como as features são do município, a média das probabilidades previstas para os alunos de uma
cidade é uma estimativa da taxa de alfabetização daquela cidade. Comparada com a meta pactuada,
ela indica quem tende a ficar abaixo.

A estimativa responde a duas perguntas, com exigências diferentes:

- **Quem fica abaixo da meta?** Exige que a probabilidade esteja certa em valor, ou a taxa
  prevista de todo município sairia deslocada. Os modelos não usam class_weight, e as
  probabilidades saem calibradas de origem (ver `calibracao`).
- **Por onde começar?** Exige só a ordem. A precisão entre os municípios de maior risco mede
  isso.

Este módulo é um backtest. O modelo usa contexto de 2023 para prever 2024, então a pergunta
"quem não atingirá a meta de 2024" pode ser respondida e, em seguida, conferida contra o que de
fato aconteceu. Para projetar a meta de 2025, o mesmo procedimento se aplica com o contexto de
2024 — o backtest é o que dá confiança para usá-lo adiante.

Avalia apenas municípios do conjunto de teste, que o modelo nunca viu no treino.

Uso: python -m src.evaluation.metas_municipios [nome_do_modelo]
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.evaluation.avaliacao import PASTA_RELATORIOS
from src.preprocessing.pipeline import CAMINHO_GOLD, preparar_dados
from src.visualization.graficos import COR, _salvar

MODELO_PADRAO = "hist_gradient_boosting"

# Município com poucos alunos avaliados tem taxa observada instável: com 15 alunos, um a mais
# alfabetizado move a taxa em quase 7 pontos. O piso evita conclusões sobre ruído.
MIN_ALUNOS = 30


def taxas_por_municipio(dados, proba_risco) -> pd.DataFrame:
    """Taxa prevista, taxa observada e meta de cada município do teste.

    As probabilidades são de RISCO (não alfabetizado); a taxa de alfabetização é o complemento.
    """
    contexto = pd.read_parquet(CAMINHO_GOLD, columns=["mun_meta_ano_alvo", "mun_elegivel_meta"])
    base = pd.DataFrame({
        "id_municipio": dados.g_teste.values,
        "sigla_uf": dados.X_teste["sigla_uf"].values,
        "regiao": dados.X_teste["regiao"].values,
        "risco_observado": dados.y_teste.values,
        "risco_previsto": proba_risco,
    }, index=dados.y_teste.index).join(contexto)

    municipios = base.groupby("id_municipio").agg(
        sigla_uf=("sigla_uf", "first"),
        regiao=("regiao", "first"),
        alunos=("risco_observado", "size"),
        taxa_prevista=("risco_previsto", lambda s: 100 * (1 - s.mean())),
        taxa_observada=("risco_observado", lambda s: 100 * (1 - s.mean())),
        meta=("mun_meta_ano_alvo", "first"),
        elegivel_meta=("mun_elegivel_meta", "first"),
    )
    return municipios.reset_index()


def classificar(municipios: pd.DataFrame) -> pd.DataFrame:
    """Separa quem tem meta comparável e marca previsto e observado em relação a ela."""
    avaliaveis = municipios[
        municipios["elegivel_meta"].fillna(False).astype(bool)
        & municipios["meta"].notna()
        & (municipios["sigla_uf"] != "RS")        # anomalia de origem documentada na Fase 2
        & (municipios["alunos"] >= MIN_ALUNOS)
    ].copy()

    avaliaveis["distancia_prevista"] = avaliaveis["taxa_prevista"] - avaliaveis["meta"]
    avaliaveis["distancia_observada"] = avaliaveis["taxa_observada"] - avaliaveis["meta"]
    avaliaveis["previsto_abaixo"] = avaliaveis["distancia_prevista"] < 0
    avaliaveis["observado_abaixo"] = avaliaveis["distancia_observada"] < 0
    return avaliaveis.sort_values("distancia_prevista")


def metricas_backtest(avaliaveis: pd.DataFrame) -> dict:
    """Acerto da previsão "ficará abaixo da meta", conferida contra o resultado de 2024."""
    y, p = avaliaveis["observado_abaixo"], avaliaveis["previsto_abaixo"]
    erro = (avaliaveis["taxa_prevista"] - avaliaveis["taxa_observada"]).abs()

    # Um gestor não usa a lista inteira: prioriza o topo. Precisão nos N primeiros mede se o
    # ranking acerta onde importa. `avaliaveis` já vem ordenado pela distância prevista.
    topo = {f"precisao_top_{n}": float(avaliaveis["observado_abaixo"].head(n).mean())
            for n in (10, 25, 50, 100) if len(avaliaveis) >= n}

    return {
        "municipios_avaliados": len(avaliaveis),
        "observados_abaixo_da_meta": int(y.sum()),
        "previstos_abaixo_da_meta": int(p.sum()),
        "acuracia": accuracy_score(y, p),
        "precisao": precision_score(y, p),   # dos sinalizados, quantos de fato ficaram abaixo
        "recall": recall_score(y, p),        # dos que ficaram abaixo, quantos foram sinalizados
        "f1": f1_score(y, p),
        "erro_medio_taxa_pontos": float(np.average(erro, weights=avaliaveis["alunos"])),
        "correlacao_prevista_observada": float(
            avaliaveis["taxa_prevista"].corr(avaliaveis["taxa_observada"])),
        **topo,
    }


def plot_taxas(avaliaveis: pd.DataFrame, nome_arquivo: str):
    """Taxa prevista x observada por município, colorida pelo acerto quanto à meta."""
    acerto = avaliaveis["previsto_abaixo"] == avaliaveis["observado_abaixo"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(avaliaveis.loc[acerto, "taxa_prevista"], avaliaveis.loc[acerto, "taxa_observada"],
               s=12, alpha=0.5, color="#1F77B4", label="acertou a situação frente à meta")
    ax.scatter(avaliaveis.loc[~acerto, "taxa_prevista"], avaliaveis.loc[~acerto, "taxa_observada"],
               s=12, alpha=0.7, color=COR, label="errou a situação frente à meta")
    ax.plot([0, 100], [0, 100], "--", color="gray", label="previsão perfeita")
    ax.set_xlabel("Taxa de alfabetização prevista (%)")
    ax.set_ylabel("Taxa de alfabetização observada em 2024 (%)")
    ax.set_title("Backtest por município — teste, sem RS")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _salvar(fig, nome_arquivo); plt.show()


def metas_municipios(nome_modelo: str = MODELO_PADRAO, dados=None):
    dados = dados or preparar_dados()

    arquivo = PASTA_RELATORIOS / f"probas_teste_{nome_modelo}.parquet"
    if not arquivo.exists():
        raise SystemExit(f"{arquivo.name} ausente. Rode antes o notebook do modelo.")
    proba = pd.read_parquet(arquivo)["proba"]
    assert proba.index.equals(dados.y_teste.index), "probabilidades de outro conjunto de teste"

    avaliaveis = classificar(taxas_por_municipio(dados, proba.values))
    metricas = metricas_backtest(avaliaveis)

    pd.Series(metricas, name=nome_modelo).to_frame().to_csv(
        PASTA_RELATORIOS / "metas_backtest.csv", encoding="utf-8-sig")
    avaliaveis.to_csv(PASTA_RELATORIOS / "metas_municipios_risco.csv", index=False,
                      encoding="utf-8-sig")
    plot_taxas(avaliaveis, "metas_backtest_municipios.png")
    return metricas, avaliaveis


if __name__ == "__main__":
    modelo = sys.argv[1] if len(sys.argv) > 1 else MODELO_PADRAO
    metricas, avaliaveis = metas_municipios(modelo)

    print(f"=== Backtest da meta de 2024 — {modelo} ===")
    for chave, valor in metricas.items():
        print(f"  {chave:34} {valor:.3f}" if isinstance(valor, float) else f"  {chave:34} {valor}")

    colunas = ["id_municipio", "sigla_uf", "alunos", "taxa_prevista", "meta", "distancia_prevista",
               "taxa_observada"]
    print("\n=== Maior risco de ficar abaixo da meta (previsão) ===")
    print(avaliaveis[colunas].head(10).round(1).to_string(index=False))
