"""
Shared text-preprocessing utilities for the chatbot.

Centralising this logic means training and inference are always using the
exact same tokenisation / lemmatisation pipeline, which avoids silent
mismatches between the bag-of-words vector built at train time and the one
built at chat time.
"""
import json
import os

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

IGNORE_TOKENS = {"?", "!", ".", "/", "@"}

lemmatizer = WordNetLemmatizer()


def ensure_nltk_data():
    """Download the NLTK corpora we need, skipping quietly if already present."""
    for package in ("punkt", "punkt_tab", "wordnet", "stopwords"):
        try:
            nltk.download(package, quiet=True)
        except Exception:
            # punkt_tab doesn't exist in every NLTK version - that's fine.
            pass


def load_intents(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find intents file at '{path}'. "
            "Pass the correct path with --data."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_and_lemmatize(sentence):
    """Tokenise a sentence and lemmatise each token, dropping punctuation."""
    tokens = nltk.word_tokenize(sentence)
    return [
        lemmatizer.lemmatize(token.lower())
        for token in tokens
        if token not in IGNORE_TOKENS
    ]


def bag_of_words(sentence, vocabulary):
    """Turn a sentence into a fixed-length binary bag-of-words vector."""
    sentence_words = set(tokenize_and_lemmatize(sentence))
    return np.array([1 if w in sentence_words else 0 for w in vocabulary], dtype=np.float32)


def build_training_vocab(intents):
    """
    Walk every pattern in the intents file and build:
      - words: sorted vocabulary of lemmatised tokens
      - classes: sorted list of intent tags
      - documents: list of (tokenised_pattern, tag) pairs
    """
    words, classes, documents = [], [], []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = sorted(set(
        lemmatizer.lemmatize(w.lower()) for w in words if w not in IGNORE_TOKENS
    ))
    classes = sorted(set(classes))
    return words, classes, documents
