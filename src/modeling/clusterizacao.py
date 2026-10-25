"""Quais regiões possuem padrões semelhantes — pergunta de negócio do enunciado.

Agrupa os municípios pelo contexto socioeconômico, educacional e territorial, com K-Means, e
depois descreve cada grupo pelo resultado de alfabetização.

**O resultado não entra no agrupamento.** `mun_taxa_ano_anterior` e
`mun_distancia_meta_anterior` medem desempenho, não contexto: incluí-las faria os grupos se
formarem em parte pelo próprio resultado, e a conclusão "contextos parecidos têm resultados
parecidos" viraria tautologia. Agrupar pelo contexto e descrever pelo resultado separa "quem são"
de "como se saem" — e permite ver municípios de contexto semelhante com desempenhos diferentes,
que é onde a política pública tem algo a aprender.

Aprendizado não supervisionado: usa todos os municípios, sem divisão treino/teste, porque não
há rótulo sendo previsto. A taxa de alfabetização é usada só depois, para descrever.

Uso: python -m src.modeling.clusterizacao
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

from src.evaluation.avaliacao import PASTA_RELATORIOS
from src.preprocessing.pipeline import CAMINHO_GOLD, RANDOM_STATE, carregar_decisoes
from src.visualization.graficos import _salvar

# Medem desempenho, não contexto — ficam fora do agrupamento (ver docstring)
EXCLUIDAS_DO_AGRUPAMENTO = ["mun_taxa_ano_anterior", "mun_distancia_meta_anterior"]

# Razões que ultrapassam 100% por defeito herdado da Fase 2: o numerador conta matrículas de
# todas as etapas, e o denominador só as de anos iniciais. Em 23,5% dos municípios o
# `mun_pct_rural` passa de 100%, com máximo de 552%. A variável continua crescendo com a
# ruralidade — serve como índice, não como percentual.
#
# Para K-Means, que usa distância euclidiana, os extremos distorceriam os grupos. O logaritmo
# preserva a ordem e reduz o peso dos valores absurdos. A correção na origem é pendência à parte.
RAZOES_SEM_TETO = ["mun_pct_rural", "mun_pct_transporte"]

FAIXA_K = range(3, 9)

# Variáveis que dão nome aos grupos na tabela de perfil
PERFIL = [
    "mun_taxa_alfabetizacao_adulta",
    "mun_pib_per_capita",
    "mun_indice_infraestrutura",
    "mun_pct_rural",
    "mun_alunos_por_turma",
    "mun_pct_integral",
    "mun_densidade",
]


def carregar_municipios():
    """Uma linha por município, com as features de contexto e a taxa observada."""
    decisoes = carregar_decisoes()
    log = [c for c in decisoes["numericas_log"] if c not in EXCLUIDAS_DO_AGRUPAMENTO]
    padrao = [c for c in decisoes["numericas_padrao"]
              if c not in EXCLUIDAS_DO_AGRUPAMENTO and c not in RAZOES_SEM_TETO]
    log += [c for c in RAZOES_SEM_TETO if c in decisoes["numericas_padrao"]]

    colunas = ["id_municipio", "sigla_uf", "regiao", "alfabetizado"] + log + padrao
    df = pd.read_parquet(CAMINHO_GOLD, columns=colunas)

    # Features são idênticas dentro do município: basta a primeira ocorrência
    municipios = df.groupby("id_municipio").agg(
        sigla_uf=("sigla_uf", "first"),
        regiao=("regiao", "first"),
        alunos=("alfabetizado", "size"),
        taxa_alfabetizacao=("alfabetizado", lambda s: 100 * s.mean()),
        **{c: (c, "first") for c in log + padrao},
    ).reset_index()
    return municipios, log, padrao


def criar_preparo(log, padrao) -> ColumnTransformer:
    """Imputação, log nas variáveis assimétricas e padronização.

    K-Means usa distância euclidiana: sem padronizar, população (milhões) dominaria proporções
    (0 a 100) e o agrupamento seria só por tamanho de cidade.
    """
    return ColumnTransformer([
        ("log", make_pipeline(SimpleImputer(strategy="median"),
                              FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                              StandardScaler()), log),
        ("padrao", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), padrao),
    ])


def escolher_k(X) -> pd.DataFrame:
    """Silhueta e inércia para cada k. A silhueta decide; a inércia documenta o cotovelo."""
    linhas = []
    for k in FAIXA_K:
        modelo = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        linhas.append({"k": k, "silhueta": silhouette_score(X, modelo.labels_),
                       "inercia": modelo.inertia_})
    return pd.DataFrame(linhas)


def perfilar(municipios: pd.DataFrame) -> pd.DataFrame:
    """Média do contexto e taxa de alfabetização de cada grupo.

    A taxa é ponderada pelos alunos e exclui o RS, cuja anomalia de origem desloca o resultado
    do estado inteiro. O contexto do RS entra normalmente — a anomalia está no resultado.
    """
    sem_rs = municipios[municipios["sigla_uf"] != "RS"]
    taxa = sem_rs.groupby("grupo").apply(
        lambda g: np.average(g["taxa_alfabetizacao"], weights=g["alunos"]))

    perfil = municipios.groupby("grupo").agg(
        municipios=("id_municipio", "size"),
        alunos=("alunos", "sum"),
        **{c: (c, "median") for c in PERFIL},
    )
    perfil.insert(2, "taxa_alfabetizacao", taxa)
    # A taxa adulta vem do Censo na escala 0 a 1; as demais taxas estão em 0 a 100
    perfil["mun_taxa_alfabetizacao_adulta"] *= 100
    return perfil.sort_values("taxa_alfabetizacao")


def distribuicao_regional(municipios: pd.DataFrame) -> pd.DataFrame:
    """Percentual dos municípios de cada região em cada grupo.

    Responde diretamente à pergunta: regiões que se distribuem pelos mesmos grupos têm padrões
    semelhantes — e um grupo que atravessa regiões mostra contextos parecidos em territórios
    diferentes.
    """
    return (pd.crosstab(municipios["regiao"], municipios["grupo"], normalize="index") * 100)


def plot_silhueta(escolha: pd.DataFrame, k: int, nome_arquivo: str):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
    eixos[0].plot(escolha["k"], escolha["silhueta"], "o-", color="#D30F59")
    eixos[0].axvline(k, color="gray", linestyle="--", label=f"k escolhido = {k}")
    eixos[0].set_xlabel("Número de grupos (k)"); eixos[0].set_ylabel("Silhueta")
    eixos[0].set_title("Silhueta — maior é melhor"); eixos[0].legend(); eixos[0].grid(alpha=0.3)
    eixos[1].plot(escolha["k"], escolha["inercia"], "o-", color="#1F77B4")
    eixos[1].axvline(k, color="gray", linestyle="--")
    eixos[1].set_xlabel("Número de grupos (k)"); eixos[1].set_ylabel("Inércia")
    eixos[1].set_title("Cotovelo"); eixos[1].grid(alpha=0.3)
    _salvar(fig, nome_arquivo); plt.show()


def plot_pca(X, municipios: pd.DataFrame, nome_arquivo: str):
    """Projeção em duas dimensões, só para visualizar — o agrupamento usa todas as variáveis."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for grupo in sorted(municipios["grupo"].unique()):
        m = (municipios["grupo"] == grupo).values
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.5, label=f"grupo {grupo}")
    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Componente 1 ({var[0]:.0f}% da variância)")
    ax.set_ylabel(f"Componente 2 ({var[1]:.0f}% da variância)")
    ax.set_title("Municípios agrupados pelo contexto (projeção PCA)")
    ax.legend(markerscale=2, fontsize=8); ax.grid(alpha=0.3)
    _salvar(fig, nome_arquivo); plt.show()


def clusterizar():
    municipios, log, padrao = carregar_municipios()
    X = criar_preparo(log, padrao).fit_transform(municipios[log + padrao])

    escolha = escolher_k(X)
    k = int(escolha.loc[escolha["silhueta"].idxmax(), "k"])

    modelo = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    municipios["grupo"] = modelo.labels_

    perfil = perfilar(municipios)
    # Renumera pela taxa, do menor para o maior desempenho — legível na apresentação
    nova_ordem = {antigo: novo for novo, antigo in enumerate(perfil.index, start=1)}
    municipios["grupo"] = municipios["grupo"].map(nova_ordem)
    perfil.index = perfil.index.map(nova_ordem)
    regional = distribuicao_regional(municipios)

    escolha.to_csv(PASTA_RELATORIOS / "clusters_silhueta.csv", index=False, encoding="utf-8-sig")
    perfil.to_csv(PASTA_RELATORIOS / "clusters_perfil.csv", encoding="utf-8-sig")
    regional.to_csv(PASTA_RELATORIOS / "clusters_regiao.csv", encoding="utf-8-sig")
    municipios[["id_municipio", "sigla_uf", "regiao", "grupo", "alunos", "taxa_alfabetizacao"]].to_csv(
        PASTA_RELATORIOS / "clusters_municipios.csv", index=False, encoding="utf-8-sig")

    plot_silhueta(escolha, k, "clusters_silhueta.png")
    plot_pca(X, municipios, "clusters_pca.png")
    return k, escolha, perfil, regional


if __name__ == "__main__":
    k, escolha, perfil, regional = clusterizar()
    print("=== Escolha de k ===")
    print(escolha.round(3).to_string(index=False))
    print(f"\n=== Perfil dos {k} grupos (medianas; taxa ponderada, sem RS) ===")
    print(perfil.round(1).to_string())
    print("\n=== % dos municípios de cada região em cada grupo ===")
    print(regional.round(0).to_string())
