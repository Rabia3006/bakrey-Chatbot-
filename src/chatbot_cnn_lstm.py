"""
Run the chatbot interactively using the trained CNN+LSTM model.

This model expects padded integer sequences (not bag-of-words), so it
needs its own preprocessing path that matches train_cnn_lstm.py exactly.

Usage:
    python src/chatbot_cnn_lstm.py
"""
import argparse
import pickle
import random

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nlp_utils import ensure_nltk_data, load_intents, lemmatizer, IGNORE_TOKENS
import nltk

ERROR_THRESHOLD = 0.25


def sentence_to_sequence(sentence, word_index, max_len):
    tokens = [
        lemmatizer.lemmatize(t.lower())
        for t in nltk.word_tokenize(sentence)
        if t not in IGNORE_TOKENS
    ]
    seq = [word_index[t] for t in tokens if t in word_index]
    return pad_sequences([seq], maxlen=max_len, padding="post")


def predict_class(sentence, model, word_index, classes, max_len):
    seq = sentence_to_sequence(sentence, word_index, max_len)
    res = model.predict(seq, verbose=0)[0]
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i], "probability": str(r)} for i, r in results]


def get_response(predictions, intents_json):
    if not predictions:
        return "Sorry, I'm not sure I understand. Could you rephrase that?"
    tag = predictions[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I'm not sure I understand. Could you rephrase that?"


def main():
    parser = argparse.ArgumentParser(description="Chat with the trained CNN+LSTM bot.")
    parser.add_argument("--model", default="models/chatbot_model_cnn_lstm.keras")
    parser.add_argument("--word-index", default="models/word_index.pkl")
    parser.add_argument("--classes", default="models/classes_cnn_lstm.pkl")
    parser.add_argument("--max-len-file", default="models/max_len.pkl")
    parser.add_argument("--data", default="data/intents.json")
    args = parser.parse_args()

    ensure_nltk_data()
    intents = load_intents(args.data)
    word_index = pickle.load(open(args.word_index, "rb"))
    classes = pickle.load(open(args.classes, "rb"))
    max_len = pickle.load(open(args.max_len_file, "rb"))
    model = load_model(args.model)

    print("Tipsybot (CNN+LSTM) is here! Type 'quit' to exit.")
    while True:
        message = input("You: ").strip()
        if message.lower() in ("quit", "exit"):
            print("Tipsybot: Goodbye!")
            break
        if not message:
            continue
        predictions = predict_class(message, model, word_index, classes, max_len)
        print(f"Tipsybot: {get_response(predictions, intents)}")


if __name__ == "__main__":
    main()
