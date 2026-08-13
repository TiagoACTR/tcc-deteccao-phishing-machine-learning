# -*- coding: utf-8 -*-
"""Avaliação de desempenho computacional (tempo e memória) dos modelos."""

import copy
import os
import time
from typing import Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
from sklearn.metrics import f1_score


def avaliar_desempenho_real(nome: str, modelo, X_train, y_train, X_test, y_test) -> dict:
    """Mede tempo de treino/predição e consumo de memória (RSS) de um modelo."""
    processo = psutil.Process(os.getpid())

    mem_inicio_rss = processo.memory_info().rss

    t0 = time.time()
    modelo.fit(X_train, y_train)
    t1 = time.time()

    mem_treino_rss = processo.memory_info().rss

    t2 = time.time()
    y_pred = modelo.predict(X_test)
    t3 = time.time()

    mem_pred_rss = processo.memory_info().rss

    uso_mem_treino = (mem_treino_rss - mem_inicio_rss) / (1024 * 1024)
    uso_mem_pred = (mem_pred_rss - mem_treino_rss) / (1024 * 1024)

    return {
        "Modelo": nome,
        "Tempo Treino (s)": t1 - t0,
        "Tempo Pred (s)": t3 - t2,
        "Memória Treino (MB)": uso_mem_treino,
        "Memória Pred (MB)": uso_mem_pred,
        "F1-Score": f1_score(y_test, y_pred),
    }


def run_performance_evaluation(modelos: Dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Executa a avaliação de desempenho computacional para todos os modelos."""
    avaliacoes = [
        avaliar_desempenho_real(nome, copy.deepcopy(modelo), X_train, y_train, X_test, y_test)
        for nome, modelo in modelos.items()
    ]
    return pd.DataFrame(avaliacoes).sort_values("F1-Score", ascending=False)


def plot_performance_comparison(df_comp: pd.DataFrame) -> None:
    """Plota tempo de treino e memória utilizada por modelo, lado a lado."""
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(data=df_comp, x="Modelo", y="Tempo Treino (s)", palette="Blues_d")
    plt.title("Tempo de Treinamento por Modelo")
    plt.ylabel("Segundos")
    plt.xlabel("")

    plt.subplot(1, 2, 2)
    sns.barplot(data=df_comp, x="Modelo", y="Memória Treino (MB)", palette="Greens_d")
    plt.title("Memória Utilizada no Treinamento por Modelo")
    plt.ylabel("Megabytes (MB)")
    plt.xlabel("")

    plt.tight_layout()
    plt.show()
