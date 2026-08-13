# -*- coding: utf-8 -*-
"""
Pré-processamento de texto para o dataset de phishing.

Contém o carregamento do modelo de linguagem (spaCy), a lista de
stopwords customizada e a função de limpeza/lematização usada para
preparar os e-mails antes da vetorização TF-IDF.
"""

import re
import sys
import subprocess

import spacy
from spacy.lang.en.stop_words import STOP_WORDS


CUSTOM_STOPWORDS = {
    'enron', 'aug', 'email', 'one', 'ect', 'time', 'submissionid', 'would',
    'message', 'note', 'may', 'submission', 'company', 'com', 'cnn', 'cnncom',
    'news', 'sender', 'subject', 'good', 'add', 'list', 'hou', 'university',
    'total', 'file', 'say', 'thank', 'year', 'daily', 'cable'
}

MAX_TEXT_LENGTH = 200_000  # limite de caracteres processados por e-mail


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Carrega o modelo do spaCy, baixando-o automaticamente se não estiver instalado."""
    try:
        nlp = spacy.load(model_name, disable=["ner", "parser"])
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name], check=False
        )
        nlp = spacy.load(model_name, disable=["ner", "parser"])

    nlp.max_length = 5_000_000
    return nlp


def build_stopwords() -> set:
    """Combina as stopwords padrão do spaCy com a lista customizada do projeto."""
    return {w.lower() for w in STOP_WORDS.union(CUSTOM_STOPWORDS)}


def preprocess_text(text: str, nlp, stopwords: set) -> str:
    """
    Limpa, tokeniza e lematiza um texto de e-mail.

    Etapas: remove caracteres não alfabéticos, converte para minúsculas,
    tokeniza/lematiza com spaCy e remove stopwords e tokens muito curtos.
    """
    text = str(text)

    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()

    doc = nlp(text)

    tokens = [
        token.lemma_.lower() for token in doc
        if token.lemma_.lower() not in stopwords
        and len(token.lemma_) > 2
        and token.is_alpha
    ]

    return " ".join(tokens)
