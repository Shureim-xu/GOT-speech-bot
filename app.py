# app.py - GoT Chatbot with Text + Audio-only WebRTC Speech Input
import streamlit as st
st.set_page_config(page_title="GoT Speech Chatbot", page_icon="🐺")

import time
import io
import numpy as np
import nltk
import string
import speech_recognition as sr
import soundfile as sf

from streamlit_webrtc import webrtc_streamer, WebRtcMode
from av import AudioFrame
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK FIXES
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Fix stopwords loading on Windows
nltk.download("stopwords", quiet=False)

try:
    _stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    _stop_words = set(stopwords.words("english"))

# LOAD KNOWLEDGE BASE
KB_FILE = "got_knowledge_base.txt"

@st.cache_data
def load_kb(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().replace("\n", " ")
        return raw
    except FileNotFoundError:
        st.error(f"Knowledge base not found at: {path}")
        st.stop()

raw_data = load_kb(KB_FILE)
sentences = sent_tokenize(raw_data)

# PREPROCESSING & CHATBOT
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = word_tokenize(text)
    return [
        _lemmatizer.lemmatize(w.lower())
        for w in words
        if w.lower() not in _stop_words and w not in string.punctuation
    ]

corpus = [preprocess(s) for s in sentences]
original_sentences = sentences.copy()

special_answers = {
    "jaime": "Jaime Lannister is a knight of the Kingsguard known as the Kingslayer.",
    "jamie": "Jaime Lannister is a knight of the Kingsguard known as the Kingslayer.",
    "jon snow": "Jon Snow is Aegon Targaryen, rightful heir to the Iron Throne.",
    "dragons": "Daenerys' dragons are Drogon, Rhaegal, and Viserion.",
    "arya": "Arya Stark is a trained assassin who kills the Night King.",
    "the wall": "The Wall protects the realms of men and is guarded by the Night's Watch.",
    "starks": "The Starks are the ruling family of Winterfell.",
}

_special_tokens = {k: set(preprocess(k)) for k in special_answers}


def get_best_sentence(query):
    q_tokens = set(preprocess(query))
    q_lower = query.lower()

    # Special cases
    for key, ans in special_answers.items():
        if key in q_lower or _special_tokens[key].intersection(q_tokens):
            return ans

    # Similarity search
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

    if best_sentence:
        return best_sentence

    return "I'm not sure — try asking about characters, houses, dragons, or major events."


def chatbot(question):
    return get_best_sentence(question)

# AUDIO RECORDER (AUDIO ONLY)
class Recorder:
    def __init__(self):
        self.frames = []
        self.sample_rate = None

    def recv_audio(self, frame: AudioFrame) -> AudioFrame:
        arr = frame.to_ndarray()
        if self.sample_rate is None:
            self.sample_rate = frame.sample_rate
        self.frames.append(arr)
        return frame

# FORCE AUDIO ONLY — FIX CAMERA ACTIVATION
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

MEDIA_CONSTRAINTS = {
    "audio": True,
    "video": False
}

# UI
st.title("🐺 Game of Thrones Chatbot - Text + Speech Input")

mode = st.radio("Choose input method:", ["Text", "Speech (microphone)"])

# TEXT MODE
if mode == "Text":
    user_text = st.text_input("Type your question:")
    if st.button("Ask"):
        if user_text.strip():
            answer = chatbot(user_text)
            st.subheader("🐺 Chatbot:")
            st.write(answer)

# SPEECH MODE – AUDIO ONLY
else:
    st.markdown("### 🎤 Speak using your microphone")

    webrtc_ctx = webrtc_streamer(
        key="got-webrtc",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=MEDIA_CONSTRAINTS,  # <<<<<<<<<<<<< FIXES CAMERA ISSUE
        audio_receiver_size=1024,
        audio_processor_factory=Recorder,
        async_processing=True,
    )

    if st.button("Transcribe & Ask"):
        if not webrtc_ctx or not webrtc_ctx.audio_processor:
            st.warning("Recorder not ready. Allow microphone & press the ▶ button.")
        else:
            proc = webrtc_ctx.audio_processor
            frames = proc.frames
            sr_rate = proc.sample_rate

            if not frames:
                st.warning("No audio captured. Press ▶ and speak.")
            else:
                # Combine frames
                arr = np.concatenate(frames, axis=1)

                # Convert to mono
                if arr.ndim == 2:
                    mono = np.mean(arr, axis=0)
                else:
                    mono = arr

                mono = mono.astype(np.float32)

                # Create WAV in memory
                bio = io.BytesIO()
                sf.write(bio, mono.T, samplerate=sr_rate, format="WAV")
                bio.seek(0)

                # Transcribe
                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(bio) as source:
                        audio_data = recognizer.record(source)
                        st.info("Transcribing...")
                        transcription = recognizer.recognize_google(audio_data)
                except:
                    transcription = "Sorry, I could not understand the audio."

                st.subheader("🗣 You said:")
                st.write(transcription)

                # Chatbot answer
                answer = chatbot(transcription)
                st.subheader("🐺 Chatbot:")
                st.write(answer)

                proc.frames = []
                proc.sample_rate = None
