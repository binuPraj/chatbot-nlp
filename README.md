# Chatbot-NLP

A simple **Natural Language Processing (NLP) chatbot** built using Python.  
It can answer questions and interact with users based on pre-defined intents and datasets.

## Features

- Responds to user inputs using NLP.
- Uses pre-trained datasets (`intents.json`, `words.pkl`, `classes.pkl`).
- Text-to-speech enabled ( using `chatbotpyttsx3.py`).
- Easy to extend with new intents and responses.

## Datasets
- `intents.json` → contains user intents and responses.
- `words.pkl` and `classes.pkl` → pickled files used for NLP model input/output.
- `best_chatbot.pth` → trained PyTorch model for NLP predictions.

> Note: The datasets are pre-trained and included in this repo.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/binuPraj/chatbot-nlp.git
cd chatbot-nlp
