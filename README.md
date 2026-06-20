# Tipsybot — Bakery Intent-Classification Chatbot

A small NLP chatbot that classifies user messages into intents (greetings,
menu questions, pricing, delivery, etc.) and replies with a matching
canned response. Two model architectures are included so you can compare
them directly.

## Project structure
bakrey-Chatbot-/

├── data/

│   └── intents.json          # training intents + responses

├── models/                   # generated at train time (gitignored)

├── src/

│   ├── nlp_utils.py          # shared tokenize/lemmatize/bag-of-words helpers

│   ├── train.py              # Dense model trainer

│   ├── train_cnn_lstm.py     # CNN+LSTM model trainer

│   ├── chatbot.py            # interactive chat loop for the Dense model

│   └── chatbot_cnn_lstm.py   # interactive chat loop for the CNN+LSTM model

├── requirements.txt

└── README.md

## Setup

This project needs Python 3.10-3.12 (TensorFlow does not yet support 3.13+
on most platforms, though some builds like 2.21 do work on 3.13 - if
something is already installed and working, prefer it over reinstalling).
conda create -n bakery python=3.11 -y

conda activate bakery

pip install --upgrade pip

pip install tensorflow nltk numpy

## Train

From the project root:
python src/train.py

python src/train_cnn_lstm.py

This writes the trained model + vocab files into models/:
- Dense model: chatbot_model.keras, words.pkl, classes.pkl
- CNN+LSTM model: chatbot_model_cnn_lstm.keras, word_index.pkl, classes_cnn_lstm.pkl, max_len.pkl

## Chat
python src/chatbot.py

python src/chatbot_cnn_lstm.py

Type a message and Tipsybot will reply based on the predicted intent. Type quit to exit.

## Dense vs CNN+LSTM

| | Dense | CNN+LSTM |
|---|---|---|
| Input representation | Binary bag-of-words vector | Padded sequence of word indices |
| Architecture | Dense(128) -> Dense(64) -> Dense(num_classes) | Embedding -> Conv1D -> MaxPooling1D -> LSTM -> Dense |
| Sensitive to word order | No | Yes |
| Data needed to perform well | Small datasets are fine | Generally needs more examples per intent |

With the current intents file (a few dozen short patterns per intent), the Dense model is a safe default. The CNN+LSTM model has more capacity to capture word order and context, but with this little training data it can also misfire. In quick manual testing, the CNN+LSTM model correctly identified "can i order customized donuts?" as a pricing-style question, while the Dense model classified the same input as a greeting.

## What changed from the original version

- Removed hardcoded absolute Windows paths.
- Split duplicated tokenize/lemmatize/bag-of-words logic into nlp_utils.py.
- Removed conflicting duplicate SGD import.
- Fixed the CNN+LSTM script: it previously fed a 2D bag-of-words vector directly into Conv1D, which has no sequence axis. It now builds padded integer sequences with a real Embedding layer.
- Added .gitignore so trained model artifacts aren't committed.
- Fixed requirements.txt (it previously listed json and pickle as pip packages, which are stdlib).
- Added a quit command and a fallback response instead of crashing on low-confidence predictions.
