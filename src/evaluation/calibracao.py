"""Calibração das probabilidades no conjunto de teste.

O enunciado pede prever se um aluno será alfabetizado "utilizando variáveis educacionais,
territoriais e socioeconômicas". Todas essas variáveis são do município: alunos da mesma
cidade recebem o mesmo vetor de features, e o modelo estima a probabilidade de risco para um
aluno inserido em determinado contexto — não para uma criança específica.

Para essa pergunta, a métrica central é a calibração: quando o modelo diz 40% de risco, cerca
de 40% dos alunos daquele contexto não se alfabetizam? ROC-AUC mede outra coisa — a capacidade
de ordenar indivíduos — e continua reportada em `comparar_teste`, ao lado desta.

Todos os modelos usam class_weight="balanced", escolha deliberada para priorizar o recall da
classe de risco. O efeito colateral é previsível: o modelo aprende como se metade dos alunos
fosse de risco, quando a taxa real no treino é cerca de 41%, e superestima o risco em 6 a 8
pontos em todas as faixas. A ordenação não é afetada — o ROC-AUC fica idêntico.

A correção é analítica e não exige retreinar: reverter o deslocamento de prior. Treinar com
pesos balanceados equivale a assumir prior de 0,5; reescalar as chances pela razão do prior real
devolve probabilidades calibradas. Usa apenas a taxa de risco do treino — nada do teste.

Reaproveita as probabilidades já gravadas por `salvar_resultados`: nenhum modelo é retreinado.

Uso: python -m src.evaluation.calibracao
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.evaluation.avaliacao import PASTA_RELATORIOS
from src.preprocessing.pipeline import CAMINHO_GOLD, preparar_dados
from src.visualization.graficos import COR, _salvar

N_FAIXAS = 10
CORES_MODELOS = [COR, "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E"]


def faixas_de_calibracao(y, proba, n_faixas: int = N_FAIXAS) -> pd.DataFrame:
    """Risco médio previsto x risco observado, em faixas de largura fixa da probabilidade.

    Largura fixa — e não quantis — porque a pergunta é "quando o modelo diz X%, acontece X%?".
    Faixas vazias são descartadas.
    """
    limites = np.linspace(0, 1, n_faixas + 1)
    faixa = pd.cut(proba, limites, include_lowest=True)
    tabela = (pd.DataFrame({"y": np.asarray(y), "proba": proba, "faixa": faixa})
              .groupby("faixa", observed=True)
              .agg(alunos=("y", "size"), risco_previsto=("proba", "mean"), risco_observado=("y", "mean")))
    tabela["diferenca"] = tabela["risco_observado"] - tabela["risco_previsto"]
    return tabela


def erro_esperado_calibracao(tabela: pd.DataFrame) -> float:
    """ECE: média da diferença absoluta entre previsto e observado, ponderada pelos alunos da faixa."""
    return float(np.average(tabela["diferenca"].abs(), weights=tabela["alunos"]))


def resumo(y, proba) -> dict:
    """Métricas de calibração. O Brier de referência é o de prever sempre a taxa média de risco."""
    y = np.asarray(y)
    brier = brier_score_loss(y, proba)
    brier_ref = brier_score_loss(y, np.full(len(y), y.mean()))
    tabela = faixas_de_calibracao(y, proba)
    return {
        "brier": brier,
        "brier_referencia": brier_ref,
        "brier_skill": 1 - brier / brier_ref,   # fração do erro da referência que o modelo elimina
        "ece_pontos": 100 * erro_esperado_calibracao(tabela),
        "roc_auc": roc_auc_score(y, proba),
        "alunos": len(y),
    }


def decis(y, proba) -> pd.DataFrame:
    """Risco observado por decil de risco previsto: a separação entre os contextos extremos."""
    decil = pd.qcut(pd.Series(proba).rank(method="first"), 10, labels=range(1, 11))
    return (pd.DataFrame({"y": np.asarray(y), "proba": proba, "decil": decil.values})
            .groupby("decil", observed=True)
            .agg(alunos=("y", "size"), risco_previsto=("proba", "mean"), risco_observado=("y", "mean")))


def corrigir_prior(proba, prior_treino: float):
    """Remove o deslocamento introduzido por class_weight="balanced".

    Pesos balanceados equivalem a treinar com prior de 0,5. A probabilidade calibrada sai de
    odds_real = odds_modelo * prior/(1 - prior), onde prior é a taxa de risco do treino.

    Transformação monotônica: preserva a ordem dos alunos, portanto ROC-AUC, PR-AUC e a curva
    precisão-recall não mudam. O limiar escolhido para o recall-alvo continua válido, desde que
    convertido pela mesma função.
    """
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * prior_treino / (1 - prior_treino)
    return odds / (1 + odds)


def carregar_probas(dados) -> dict:
    """Probabilidades de teste de cada modelo, conferidas contra o teste atual."""
    probas = {}
    for arq in sorted(PASTA_RELATORIOS.glob("probas_teste_*.parquet")):
        nome = arq.stem.replace("probas_teste_", "")
        proba = pd.read_parquet(arq)["proba"]
        assert proba.index.equals(dados.y_teste.index), f"{nome}: teste diferente do atual"
        probas[nome] = proba.values
    return probas


def plot_calibracao(antes: dict, depois: dict, nome_arquivo: str):
    """Diagrama de confiabilidade, antes e depois da correção de prior."""
    fig, eixos = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, curvas, titulo in ((eixos[0], antes, "Original (class_weight balanced)"),
                               (eixos[1], depois, "Após correção de prior")):
        ax.plot([0, 1], [0, 1], "--", color="gray", label="calibração perfeita")
        for (nome, tabela), cor in zip(curvas.items(), CORES_MODELOS):
            ax.plot(tabela["risco_previsto"], tabela["risco_observado"], "o-", color=cor,
                    linewidth=2, markersize=5, label=nome)
        ax.set_xlabel("Risco previsto pelo modelo"); ax.set_title(titulo)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    eixos[0].set_ylabel("Risco observado no teste")
    fig.suptitle("Calibração no teste — sem RS, municípios nunca vistos no treino")
    _salvar(fig, nome_arquivo); plt.show()


# Chaves e alvo de `features_aluno`: não são features
NAO_FEATURES = ["ano", "id_aluno", "id_escola", "id_municipio", "alfabetizado", "peso_aluno"]


def features_que_variam_no_municipio() -> tuple:
    """Quantas colunas de `features_aluno` assumem mais de um valor dentro de um município."""
    df = pd.read_parquet(CAMINHO_GOLD)
    features = [c for c in df.columns if c not in NAO_FEATURES]
    variam = [c for c in features
              if (df.groupby("id_municipio")[c].nunique(dropna=False) > 1).any()]
    return variam, len(features)


def teto_municipal(dados, probas: dict, prior: float) -> pd.DataFrame:
    """O melhor resultado possível com features municipais, e quanto dele cada modelo alcança.

    Como todo aluno de uma cidade recebe o mesmo vetor de features, nenhum modelo supera um
    previsor que soubesse a taxa real de cada município. Ele não existe — usa o resultado que
    se quer prever —, mas define o teto de qualquer modelo construído com estes dados.

    "Fração alcançada" é o ganho do modelo dividido pelo ganho do teto: sobre o acaso, no
    ROC-AUC; sobre prever sempre a média, no Brier. Mesma base em tudo: teste, sem RS.
    """
    sem_rs = (dados.X_teste["sigla_uf"] != "RS").values
    y = dados.y_teste.values[sem_rs]
    grupos = dados.g_teste.values[sem_rs]

    taxa_real = pd.Series(y).groupby(grupos).transform("mean").values
    referencia = brier_score_loss(y, np.full(len(y), y.mean()))
    skill = lambda p: 1 - brier_score_loss(y, p) / referencia

    auc_teto, skill_teto = roc_auc_score(y, taxa_real), skill(taxa_real)

    anterior = dados.X_teste["mun_taxa_ano_anterior"].values[sem_rs]
    anterior = np.where(np.isnan(anterior), np.nanmedian(anterior), anterior)

    variam, total = features_que_variam_no_municipio()
    contexto = {
        "alunos": len(y),
        "municipios": len(set(grupos)),
        "features_que_variam_no_municipio": f"{len(variam)} de {total} ({', '.join(variam)})",
        # O previsor perfeito elimina exatamente a variação entre municípios
        "variancia_entre_municipios_pct": 100 * np.var(taxa_real) / np.var(y),
        "roc_auc_uma_feature_taxa_ano_anterior": roc_auc_score(y, -anterior),
        "roc_auc_teto": auc_teto,
        "brier_skill_teto_pct": 100 * skill_teto,
    }

    linhas = []
    for nome, proba in probas.items():
        p = proba[sem_rs]
        auc = roc_auc_score(y, p)
        linhas.append({
            "modelo": nome,
            "roc_auc": auc,
            "fracao_roc_auc_pct": 100 * (auc - 0.5) / (auc_teto - 0.5),
            "brier_skill_original_pct": 100 * skill(p),
            "fracao_brier_original_pct": 100 * skill(p) / skill_teto,
            "brier_skill_corrigido_pct": 100 * skill(corrigir_prior(p, prior)),
            "fracao_brier_corrigido_pct": 100 * skill(corrigir_prior(p, prior)) / skill_teto,
        })

    resultado = pd.DataFrame(linhas).sort_values("fracao_brier_corrigido_pct", ascending=False)
    pd.Series(contexto).to_frame("valor").to_csv(
        PASTA_RELATORIOS / "teto_contexto.csv", encoding="utf-8-sig")
    resultado.to_csv(PASTA_RELATORIOS / "teto_teste.csv", index=False, encoding="utf-8-sig")
    return contexto, resultado


def calibracao_teste(dados=None) -> pd.DataFrame:
    """Calibração de todos os modelos, com e sem RS.

    A versão sem RS é a principal: a anomalia documentada na Fase 2 desloca o risco observado do
    estado inteiro, e misturá-la à calibração mede a anomalia, não o modelo.
    """
    dados = dados or preparar_dados()
    probas = carregar_probas(dados)
    prior = float(dados.y_treino.mean())   # taxa de risco do TREINO
    sem_rs = (dados.X_teste["sigla_uf"] != "RS").values
    todos = np.ones_like(sem_rs, dtype=bool)

    linhas, antes, depois = [], {}, {}
    for nome, proba_original in probas.items():
        proba_corrigida = corrigir_prior(proba_original, prior)

        # Gravadas com outro prefixo para nao colidir com o glob de comparar_teste
        pd.DataFrame({"proba": proba_corrigida}, index=dados.y_teste.index).to_parquet(
            PASTA_RELATORIOS / f"probas_calibradas_{nome}.parquet")

        for versao, proba in (("original", proba_original), ("corrigida", proba_corrigida)):
            for recorte, mascara in (("sem RS", sem_rs), ("com RS", todos)):
                y, p = dados.y_teste.values[mascara], proba[mascara]
                linhas.append({"modelo": nome, "probabilidade": versao, "recorte": recorte,
                               **resumo(y, p)})
                if recorte == "sem RS":
                    tabela = faixas_de_calibracao(y, p)
                    (antes if versao == "original" else depois)[nome] = tabela
                    tabela.to_csv(PASTA_RELATORIOS / f"calibracao_{nome}_{versao}_faixas.csv",
                                  encoding="utf-8-sig")
                    if versao == "corrigida":
                        decis(y, p).to_csv(PASTA_RELATORIOS / f"calibracao_{nome}_decis.csv",
                                           encoding="utf-8-sig")

    resultado = (pd.DataFrame(linhas)
                 .sort_values(["recorte", "probabilidade", "brier"], ascending=[False, True, True]))
    resultado.to_csv(PASTA_RELATORIOS / "calibracao_teste.csv", index=False, encoding="utf-8-sig")
    plot_calibracao(antes, depois, "calibracao_teste.png")
    return resultado, teto_municipal(dados, probas, prior)


if __name__ == "__main__":
    tabela, (contexto, teto) = calibracao_teste()
    print("=== Calibração no teste ===")
    print(tabela.round(4).to_string(index=False))
    print("\n=== O limite dos dados — teste, sem RS ===")
    for chave, valor in contexto.items():
        print(f"  {chave:42} {valor:.3f}" if isinstance(valor, float) else f"  {chave:42} {valor}")
    print("\n=== Quanto do teto cada modelo alcança ===")
    print(teto.round({"roc_auc": 3}).round(1).to_string(index=False))
