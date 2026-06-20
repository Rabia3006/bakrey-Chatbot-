"""
Train a simple feed-forward (Dense) intent classifier.

Usage:
    python src/train.py
    python src/train.py --data data/intents.json --out models --epochs 200
"""
import argparse
import os
import pickle
import random

import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from nlp_utils import build_training_vocab, ensure_nltk_data, load_intents


def build_dataset(intents):
    words, classes, documents = build_training_vocab(intents)

    training = []
    output_empty = [0] * len(classes)
    for tokens, tag in documents:
        lemmas = {w.lower() for w in tokens}
        bag = [1 if w in lemmas else 0 for w in words]

        output_row = list(output_empty)
        output_row[classes.index(tag)] = 1
        training.append((bag, output_row))

    random.shuffle(training)
    train_x = np.array([row[0] for row in training], dtype=np.float32)
    train_y = np.array([row[1] for row in training], dtype=np.float32)
    return words, classes, train_x, train_y


def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, input_shape=(input_dim,), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(output_dim, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the Dense intent-classifier chatbot model.")
    parser.add_argument("--data", default="data/intents.json", help="Path to intents.json")
    parser.add_argument("--out", default="models", help="Directory to write model + vocab files")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ensure_nltk_data()
    intents = load_intents(args.data)
    words, classes, train_x, train_y = build_dataset(intents)

    pickle.dump(words, open(os.path.join(args.out, "words.pkl"), "wb"))
    pickle.dump(classes, open(os.path.join(args.out, "classes.pkl"), "wb"))

    model = build_model(input_dim=train_x.shape[1], output_dim=train_y.shape[1])
    model.fit(train_x, train_y, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    model_path = os.path.join(args.out, "chatbot_model.keras")
    model.save(model_path)
    print(f"\nTraining complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
