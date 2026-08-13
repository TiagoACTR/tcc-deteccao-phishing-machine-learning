# -*- coding: utf-8 -*-
"""Funções de análise exploratória (contagem de palavras e gráficos)."""

from collections import Counter
from itertools import chain

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_url_distribution(df: pd.DataFrame) -> None:
    """Plota a quantidade de e-mails com URL por classe (legítimo x phishing)."""
    df_urls = df[df["has_url"] == 1]
    url_counts = df_urls["label"].value_counts().sort_index()

    sns.barplot(
        x=["Legítimos", "Phishing"],
        y=url_counts.values,
        palette=["#6BAED6", "#F03B20"],
    )
    plt.title("Quantidade de e-mails que possuem URLs por classe")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de e-mails com links")
    plt.show()


def plot_class_distribution(y) -> None:
    """Plota a distribuição das classes (phishing vs. legítimo)."""
    sns.countplot(x=y)
    plt.title("Distribuição das Classes (Phishing vs. Legítimo)")
    plt.show()


def top_n_palavras(serie_textos: pd.Series, n: int = 20) -> pd.DataFrame:
    """Retorna as `n` palavras mais frequentes em uma série de textos."""
    todas = list(chain.from_iterable(serie_textos.str.split()))
    cont = Counter(todas)
    top = cont.most_common(n)
    return pd.DataFrame(top, columns=["Palavra", "Frequência"])


def plot_top_words(df_top: pd.DataFrame, title: str, palette: str) -> None:
    """Plota um gráfico de barras horizontais com as palavras mais frequentes."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_top, x="Frequência", y="Palavra", palette=palette, orient="h")
    plt.title(title)
    plt.xlabel("Frequência")
    plt.ylabel("")
    plt.show()


def run_word_frequency_analysis(df: pd.DataFrame) -> None:
    """Executa a análise de frequência de palavras para o dataset completo e por classe."""
    df_top_all = top_n_palavras(df["text_cleaned"], 20)
    df_top_phishing = top_n_palavras(df[df["label"] == 1]["text_cleaned"], 20)
    df_top_legitimos = top_n_palavras(df[df["label"] == 0]["text_cleaned"], 20)

    print("Top 20 palavras — Dataset completo")
    print(df_top_all)
    print("\nTop 20 palavras — E-mails de phishing")
    print(df_top_phishing)
    print("\nTop 20 palavras — E-mails legítimos")
    print(df_top_legitimos)

    plot_top_words(df_top_all, "Top 20 Palavras — Dataset Completo", "Blues_d")
    plot_top_words(df_top_phishing, "Top 20 Palavras — Phishing", "Reds_d")
    plot_top_words(df_top_legitimos, "Top 20 Palavras — Legítimos", "Greens_d")
