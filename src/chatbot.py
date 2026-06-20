"""
Run the chatbot interactively from the terminal using the trained Dense model.

Usage:
    python src/chatbot.py
    python src/chatbot.py --model models/chatbot_model.keras --data data/intents.json
"""
import argparse
import pickle
import random

import numpy as np
from tensorflow.keras.models import load_model

from nlp_utils import bag_of_words, ensure_nltk_data, load_intents

ERROR_THRESHOLD = 0.25


def predict_class(sentence, model, words, classes):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
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
    parser = argparse.ArgumentParser(description="Chat with the trained intent-classifier bot.")
    parser.add_argument("--model", default="models/chatbot_model.keras")
    parser.add_argument("--words", default="models/words.pkl")
    parser.add_argument("--classes", default="models/classes.pkl")
    parser.add_argument("--data", default="data/intents.json")
    args = parser.parse_args()

    ensure_nltk_data()
    intents = load_intents(args.data)
    words = pickle.load(open(args.words, "rb"))
    classes = pickle.load(open(args.classes, "rb"))
    model = load_model(args.model)

    print("Tipsybot is here! Type 'quit' to exit.")
    while True:
        message = input("You: ").strip()
        if message.lower() in ("quit", "exit"):
            print("Tipsybot: Goodbye!")
            break
        if not message:
            continue
        predictions = predict_class(message, model, words, classes)
        print(f"Tipsybot: {get_response(predictions, intents)}")


if __name__ == "__main__":
    main()
