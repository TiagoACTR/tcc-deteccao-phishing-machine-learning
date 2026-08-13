# -*- coding: utf-8 -*-
"""Validação cruzada, métricas de desempenho e comparação entre modelos."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    make_scorer, ConfusionMatrixDisplay,
)


SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, pos_label=1),
    "recall": make_scorer(recall_score, pos_label=1),
    "f1": make_scorer(f1_score, pos_label=1),
}


def get_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Cria o esquema de validação cruzada estratificada usado no projeto."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def avalia_modelo_cv(pipe, X_train, y_train, cv: StratifiedKFold) -> dict:
    """Executa a validação cruzada estratificada para um pipeline."""
    return cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)


def resumo_ic(scores) -> Tuple[float, Tuple[float, float]]:
    """Calcula a média e o intervalo de confiança de 95% para uma lista de scores."""
    n = len(scores)
    m = np.mean(scores)
    se = np.std(scores, ddof=1) / np.sqrt(n)
    z = norm.ppf(0.975)
    return m, (m - z * se, m + z * se)


def resumo_cv(nome: str, cvres: dict) -> dict:
    """Resume os resultados de validação cruzada de um modelo em um dicionário."""
    return {
        "Modelo": nome,
        "Acc (mean)": np.mean(cvres["test_accuracy"]),
        "Prec (mean)": np.mean(cvres["test_precision"]),
        "Rec (mean)": np.mean(cvres["test_recall"]),
        "F1 (mean)": np.mean(cvres["test_f1"]),
    }


def run_cross_validation(modelos: Dict, X_train, y_train) -> Dict[str, dict]:
    """Roda a validação cruzada para todos os modelos e imprime o F1 com IC95%."""
    cv = get_cv()
    resultados = {}
    for nome, pipe in modelos.items():
        cvres = avalia_modelo_cv(pipe, X_train, y_train, cv)
        resultados[nome] = cvres
        f1_m, (f1_lo, f1_hi) = resumo_ic(cvres["test_f1"])
        print(f"{nome}: F1 = {f1_m:.3f}  (IC95% {f1_lo:.3f}–{f1_hi:.3f})")
    return resultados


def build_comparison_table(resultados_cv: Dict[str, dict]) -> pd.DataFrame:
    """Monta a tabela comparativa (ordenada por F1) a partir dos resultados de CV."""
    linhas = [resumo_cv(nome, cvres) for nome, cvres in resultados_cv.items()]
    return pd.DataFrame(linhas).sort_values("F1 (mean)", ascending=False).reset_index(drop=True)


def plot_confusion_matrices(modelos: Dict, X_train, y_train, X_test, y_test) -> None:
    """Treina cada modelo no conjunto de treino completo e plota sua matriz de confusão."""
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        plt.figure(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", colorbar=False)
        plt.title(f"Matriz de Confusão - {nome}")
        plt.tight_layout()
        plt.show()


def plot_metric_comparison(df_cv: pd.DataFrame) -> None:
    """Plota um gráfico de barras comparando as métricas médias de cada modelo."""
    df_grafico = df_cv.rename(columns={
        "Acc (mean)": "Acurácia",
        "Prec (mean)": "Precisão",
        "Rec (mean)": "Recall",
        "F1 (mean)": "F1-Score",
    })

    melt = df_grafico.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melt, x="Métrica", y="Valor", hue="Modelo", palette="Blues_d")
    plt.title("Comparação de Desempenho dos Modelos (Validação Cruzada)")
    plt.ylim(0, 1)
    plt.ylabel("Valor da Métrica")
    plt.xlabel("Métrica de Desempenho")
    plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
