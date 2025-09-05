import streamlit as st
import json
import os
import pickle
import random
import spacy
import numpy as np
import torch
import torch.nn as nn
import streamlit.components.v1 as components


#-- for audio --
TTS_ENABLED = True

def speak_js(text):
    escaped_text = text.replace('"', '\\"')  # Escape double quotes
    components.html(
        f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{escaped_text}");
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )

# -- train model --
nlp = spacy.load("en_core_web_sm")

class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def clean_up_sentence(sentence):
    doc = nlp(sentence)
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence, model, words, classes):
    bow = bag_of_words(sentence, words)
    bow_tensor = torch.from_numpy(bow).float().unsqueeze(0)
    output = model(bow_tensor)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

def get_response(intent_tag, intents):
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
        

# -- train model --
def train_model(epochs, progress_bar=None):
    with open("intents.json") as file:
        intents = json.load(file)

    for file in ["words.pkl", "classes.pkl", "best_chatbot.pth"]:
        if os.path.exists(file):
            os.remove(file)

    words = []
    classes = []
    documents = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            w = clean_up_sentence(pattern)
            words.extend(w)
            documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    training = []
    for doc in documents:
        bag = [0] * len(words)
        for w in doc[0]:
            if w in words:
                bag[words.index(w)] = 1
        label = [0] * len(classes)
        label[classes.index(doc[1])] = 1
        training.append([bag, label])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    model = ChatbotModel(len(train_x[0]), 512, len(train_y[0]))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        inputs = torch.from_numpy(train_x).float()
        targets = torch.from_numpy(np.argmax(train_y, axis=1))
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if progress_bar:
            progress_bar.progress((epoch + 1) / epochs)

    torch.save(model.state_dict(), "best_chatbot.pth")
    with open("words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open("classes.pkl", "wb") as f:
        pickle.dump(classes, f)

# --- chatbot layout and functionality---
st.set_page_config(page_title="UFO Nepal Chatbot", page_icon="🧵", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        body {
            background-color: #ababab;
            color: #000000;
        }
        .stTextInput > div > input {
            background-color: #ffffff;
            color: black;
        }
        .stButton > button {
            background-color: #122442;
            color: white;
        }
        .stChatMessage {
            background-color: #202224;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧵 UFO Nepal Website Chatbot (Project)")
st.markdown("Train your chatbot and start chatting about UFO Nepal.")

if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'greeted' not in st.session_state:
    st.session_state.greeted = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.form("user_form"):
    name = st.text_input("👤 Your Name")
    epochs = st.number_input("🔄 Number of Training Epochs", min_value=1000, max_value=30000, value=1000, step=50)
    submitted = st.form_submit_button("🏋️ Train Chatbot")

if submitted:
    with st.spinner("Training your chatbot, please wait..."):
        st.session_state.username = name
        progress_bar = st.progress(0)
        train_model(epochs, progress_bar)
        st.session_state.trained = True
        st.session_state.greeted = False
        st.session_state.chat_history = []
    st.success(f"Training complete! Welcome, {name}!")

if st.session_state.trained:
    st.subheader("💬 Ask your Chatbot about UFO Nepal")

    intents = json.load(open('intents.json'))
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = ChatbotModel(len(words), 512, len(classes))
    model.load_state_dict(torch.load("best_chatbot.pth"))
    model.eval()

    if not st.session_state.greeted:
        greeting = f"Hello, {st.session_state.username}! I'm your UFO Nepal chatbot. Ask me anything."
        st.session_state.chat_history.append(("assistant", greeting))
        if TTS_ENABLED:
            speak_js(greeting)
        st.session_state.greeted = True

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        intent_tag = predict_class(user_input, model, words, classes)
        response = get_response(intent_tag, intents)
        st.session_state.chat_history.append(("assistant", response))

        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

        if TTS_ENABLED:
            speak_js(response)
