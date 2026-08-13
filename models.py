# -*- coding: utf-8 -*-
"""Definição dos pipelines de classificação avaliados no projeto."""

from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline


def build_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """Cria os três pipelines (TF-IDF + classificador) avaliados no TCC."""
    pipe_nb = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
    pipe_svm = make_pipeline(TfidfVectorizer(stop_words="english"), LinearSVC())
    pipe_rf = make_pipeline(
        TfidfVectorizer(stop_words="english"),
        RandomForestClassifier(n_estimators=100, random_state=random_state),
    )

    return {
        "MultinomialNB": pipe_nb,
        "LinearSVC": pipe_svm,
        "RandomForest": pipe_rf,
    }
