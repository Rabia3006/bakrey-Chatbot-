"""
Train a hybrid CNN + LSTM intent classifier with an embedding layer.

Unlike the Dense model in train.py, this one treats each pattern as a
*sequence of word indices* (not a bag-of-words vector), since Conv1D/LSTM
need an actual sequence dimension to operate over. That's the piece that
was missing from the original script - it tried to run Conv1D directly on
a flat bag-of-words vector, which doesn't have a sequence axis at all.

Usage:
    python src/train_cnn_lstm.py
    python src/train_cnn_lstm.py --data data/intents.json --out models --epochs 100
"""
import argparse
import os
import pickle
import random

import numpy as np
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nlp_utils import ensure_nltk_data, load_intents, lemmatizer, IGNORE_TOKENS
import nltk


def build_sequence_dataset(intents, max_len=20):
    documents = []
    classes = []
    vocab = set()

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            tokens = [
                lemmatizer.lemmatize(t.lower())
                for t in nltk.word_tokenize(pattern)
                if t not in IGNORE_TOKENS
            ]
            documents.append((tokens, intent["tag"]))
            vocab.update(tokens)
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    vocab = sorted(vocab)
    classes = sorted(classes)
    word_index = {w: i + 1 for i, w in enumerate(vocab)}  # 0 reserved for padding

    sequences, labels = [], []
    output_empty = [0] * len(classes)
    for tokens, tag in documents:
        seq = [word_index[t] for t in tokens]
        sequences.append(seq)
        row = list(output_empty)
        row[classes.index(tag)] = 1
        labels.append(row)

    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)

    padded = pad_sequences(sequences, maxlen=max_len, padding="post")
    return word_index, classes, np.array(padded), np.array(labels, dtype=np.float32), max_len


def build_model(vocab_size, max_len, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the hybrid CNN+LSTM intent-classifier.")
    parser.add_argument("--data", default="data/intents.json", help="Path to intents.json")
    parser.add_argument("--out", default="models", help="Directory to write model + vocab files")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=20, help="Max tokens per pattern (padded/truncated)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ensure_nltk_data()
    intents = load_intents(args.data)
    word_index, classes, train_x, train_y, max_len = build_sequence_dataset(intents, args.max_len)

    pickle.dump(word_index, open(os.path.join(args.out, "word_index.pkl"), "wb"))
    pickle.dump(classes, open(os.path.join(args.out, "classes_cnn_lstm.pkl"), "wb"))
    pickle.dump(max_len, open(os.path.join(args.out, "max_len.pkl"), "wb"))

    model = build_model(vocab_size=len(word_index), max_len=max_len, num_classes=len(classes))
    model.fit(train_x, train_y, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    model_path = os.path.join(args.out, "chatbot_model_cnn_lstm.keras")
    model.save(model_path)
    print(f"\nTraining complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
