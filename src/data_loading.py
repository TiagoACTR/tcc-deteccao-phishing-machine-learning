# -*- coding: utf-8 -*-
"""Carregamento e exploração inicial do dataset de e-mails de phishing."""

import pandas as pd


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV do dataset e exibe um resumo inicial no console."""
    df = pd.read_csv(csv_path)

    print("Dimensões do dataset:", df.shape)
    print("\nColunas disponíveis:", df.columns.tolist())
    print("\nValores faltantes por coluna:")
    print(df.isnull().sum())
    print("\nDistribuição da coluna 'label':")
    print(df["label"].value_counts())
    print("\nPrimeiras linhas:")
    print(df.head(5))

    return df


def add_url_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a coluna binária `has_url`, indicando presença de URL no e-mail."""
    df["has_url"] = df["text_combined"].str.contains(
        r"http\S+|www\S+", regex=True
    ).astype(int)

    print("Distribuição geral de e-mails com URLs:")
    print(df["has_url"].value_counts(normalize=True).round(3) * 100)

    print("\nProporção média de e-mails com URLs por classe:")
    print(df.groupby("label")["has_url"].mean().round(3) * 100)

    return df
