# -*- coding: utf-8 -*-
"""
Detecção de Phishing com Machine Learning
------------------------------------------
TCC — Segurança da Informação — Universidade Federal do Ceará (UFC)
Autor: Tiago Andrade | Orientador: Prof. Israel Eduardo | 2025

Ponto de entrada do projeto: orquestra o carregamento dos dados,
pré-processamento, análise exploratória, validação cruzada e
avaliação de desempenho dos modelos de classificação.
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data_loading import load_dataset, add_url_feature
from src.preprocessing import load_spacy_model, build_stopwords, preprocess_text
from src.eda import (
    plot_url_distribution,
    plot_class_distribution,
    run_word_frequency_analysis,
)
from src.models import build_models
from src.evaluation import (
    run_cross_validation,
    build_comparison_table,
    plot_confusion_matrices,
    plot_metric_comparison,
)
from src.performance import run_performance_evaluation, plot_performance_comparison


CSV_PATH = "data/phishing_email.csv"  # ajuste para o caminho local do dataset baixado do Kaggle
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    sns.set(style="whitegrid", palette="muted")
    pd.set_option("display.max_colwidth", 150)

    # 1. Carregamento e exploração inicial dos dados
    df = load_dataset(CSV_PATH)
    df = add_url_feature(df)
    plot_url_distribution(df)

    # 2. Pré-processamento de texto
    nlp = load_spacy_model()
    stopwords = build_stopwords()

    tqdm.pandas()
    df["text_cleaned"] = df["text_combined"].astype(str).progress_apply(
        lambda t: preprocess_text(t, nlp, stopwords)
    )
    print("Pré-processamento concluído!")

    X = df["text_cleaned"]
    y = df["label"].astype(int)

    # 3. Análise exploratória
    plot_class_distribution(y)
    run_word_frequency_analysis(df)

    # 4. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, shuffle=True, random_state=RANDOM_STATE
    )
    print("Treino:", len(X_train), "| Teste:", len(X_test))

    # 5. Modelagem e validação cruzada
    modelos = build_models(random_state=RANDOM_STATE)
    resultados_cv = run_cross_validation(modelos, X_train, y_train)
    df_cv = build_comparison_table(resultados_cv)
    print(df_cv)

    # 6. Teste final: matrizes de confusão e comparação de métricas
    plot_confusion_matrices(modelos, X_train, y_train, X_test, y_test)
    plot_metric_comparison(df_cv)

    # 7. Desempenho computacional
    df_comp = run_performance_evaluation(modelos, X_train, y_train, X_test, y_test)
    print(df_comp)
    plot_performance_comparison(df_comp)


if __name__ == "__main__":
    main()
