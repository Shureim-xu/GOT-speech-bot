import nltk
nltk.data.path.append("nltk_data")

import streamlit as st

# Set page config at the very top
st.set_page_config(page_title="GoT Speech Chatbot", page_icon="🐺")

import speech_recognition as sr
import nltk
import string
import time
import io
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


try:
    # Check if 'punkt' is already available
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    # If not found, download it programmatically
    nltk.download('punkt')
    
# Load knowledge base
KB_FILE = "got_knowledge_base.txt"

@st.cache_data
def load_kb(path):
    # It's good practice to ensure the KB file exists before trying to open it
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().replace("\n", " ")
        return raw
    except FileNotFoundError:
        st.error(f"Error: Knowledge base file '{path}' not found.")
        st.stop() # Stop the script if data isn't there

raw_data = load_kb(KB_FILE)
sentences = sent_tokenize(raw_data)

# Preprocessing
# FIX: The original error was right here. 'english' was not a string.
_stop_words = set(stopwords.words("english")) 
_lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(w.lower())
        for w in words
        if w.lower() not in _stop_words and w not in string.punctuation
    ]
    return tokens

corpus = [preprocess(s) for s in sentences]
original_sentences = sentences.copy()

# Special direct answers
special_answers = {
    "jaime": "Jaime Lannister is a knight of the Kingsguard known as the Kingslayer.",
    "jamie": "Jaime Lannister is a knight of the Kingsguard known as the Kingslayer.",
    "jon snow": "Jon Snow is revealed as Aegon Targaryen, the true heir to the Iron Throne.",
    "dragons": "Daenerys' dragons are Drogon, Rhaegal, and Viserion.",
    "arya": "Arya Stark is a trained assassin who kills the Night King.",
    "the wall": "The Wall protects the realms of men and is guarded by the Night's Watch.",
    "starks": "The Starks rule Winterfell and descend from the First Men.",
}

_special_tokens = {k: set(preprocess(k)) for k in special_answers}

# Chatbot logic
def get_best_sentence(query):
    q_tokens = set(preprocess(query))
    q_lower = query.lower()

    # special answers first
    for key, ans in special_answers.items():
        if key in q_lower or _special_tokens[key].intersection(q_tokens):
            return ans

    # similarity search
    best_score = 0
    best_sentence = None

    for i, s_tokens in enumerate(corpus):
        s_text = original_sentences[i]
        if "keywords" in s_text.lower():
            continue

        intersection = len(set(s_tokens).intersection(q_tokens))
        union = len(set(s_tokens).union(q_tokens))
        if union == 0:
            continue

        score = intersection / union
        if score > best_score:
            best_score = score
            best_sentence = s_text

    if best_sentence and best_score > 0:
        return best_sentence

    return "I’m not sure — try asking about characters, houses, or major events."


def chatbot(question):
    return get_best_sentence(question)


# Speech Recognition
def transcribe_speech(pause_duration=0.8): # Removed 'language' arg which wasn't used in the call below
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        r.pause_threshold = pause_duration

        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            st.info("Transcribing...")
        except sr.WaitTimeoutError:
            st.warning("No speech detected.")
            return None
        except Exception as e:
            st.error(f"Microphone error: {e}")
            return None

        try:
            # Using 'en-US' as default language code for Google recognition
            return r.recognize_google(audio, language="en-US") 
        except sr.UnknownValueError:
            st.warning("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Speech recognition service unavailable.")
            return None
        except Exception as e:
            st.error(f"Unexpected error during transcription: {e}")
            return None

# --- Streamlit UI Section ---

st.title("🐺 Game of Thrones Chatbot — Speech + Text ")
st.write("Use **text** or upload an **audio recording** to ask your question.")


# Chat history setup
if "history" not in st.session_state:
    st.session_state.history = []


mode = st.radio("Choose input method:", ["Text", "Upload Audio"])

# TEXT INPUT MODE
if mode == "Text":
    user_text = st.text_input("Type your question:")
    if st.button("Ask"):
        if user_text.strip():
            answer = chatbot(user_text)
            st.session_state.history.append(("You", user_text))
            st.session_state.history.append(("Bot", answer))

# SPEECH INPUT MODE (Moved UI elements down here for clarity)
if mode == "Upload Audio": # Original code seemed to treat this as just "microphone input"
    # The original script had UI elements for speech *outside* the mode check, I put them inside here
    pause = st.slider(
        "Pause Threshold (controls pause/continue)",
        0.5, 2.0, 0.8, 0.1
    )

    if st.button("🎙 Start Talking"):
        # The function was called with 'pause' value where 'language' should have been. 
        # Corrected the function call and definition.
        transcription = transcribe_speech(pause_duration=pause) 

        if transcription:
            st.subheader("🗣 You Said:")
            st.write(transcription)

            # Check transcription results before generating an answer
            if transcription not in ["No speech detected.", "Sorry, I could not understand the audio.", "Speech recognition service unavailable.", None]:
                answer = chatbot(transcription)
                st.subheader("🐺 Chatbot:")
                st.write(answer)
                
                st.session_state.history.append(("You (Speech)", transcription))
                st.session_state.history.append(("Bot", answer))

                # save option
                if st.button("💾 Save Chat"):
                    filename = f"chat_{int(time.time())}.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"You: {transcription}\nBot: {answer}")
                    st.success(f"Saved as {filename}")

# Display Chat History at the bottom
st.subheader("Chat History")
for role, message in reversed(st.session_state.history):
    if role == "You" or role == "You (Speech)":
        st.markdown(f"**{role}:** *{message}*")
    else:
        st.markdown(f"**{role}:** {message}")

