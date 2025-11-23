# app.py - GoT Speech Chatbot using streamlit-webrtc (WAV)
import streamlit as st
st.set_page_config(page_title="GoT Speech Chatbot (WebRTC)", page_icon="🐺")

import time
import io
import numpy as np
import nltk
import string
import speech_recognition as sr
import soundfile as sf

from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from av import AudioFrame
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK downloads (quiet)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Knowledge base path (uploaded)
KB_FILE = "/mnt/data/got_knowledge_base.txt"

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

# Preprocessing & chatbot logic
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

def get_best_sentence(query):
    q_tokens = set(preprocess(query))
    q_lower = query.lower()

    for key, ans in special_answers.items():
        if key in q_lower or _special_tokens[key].intersection(q_tokens):
            return ans

    best_score = 0.0
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

# Recorder class for webrtc
class Recorder:
    def __init__(self):
        self.frames = []  # list of numpy arrays (channels, samples)
        self.sample_rate = None

    def recv_audio(self, frame: AudioFrame) -> AudioFrame:
        # Called for each incoming audio frame by streamlit-webrtc.
        # Convert frame to ndarray (shape: channels x samples)
        arr = frame.to_ndarray()
        # store sample rate once
        if self.sample_rate is None:
            self.sample_rate = frame.sample_rate
        self.frames.append(arr)
        return frame  # pass-through

# UI
st.title("🐺 Game of Thrones - Speech Chatbot (browser recorder)")
st.write("Use the recorder below to capture audio in your browser. When finished, click **Transcribe & Ask** to send the audio to the chatbot.")

st.markdown(
    """
**Notes**
- The recorder uses the browser's microphone (WebRTC).
- No PyAudio or server-side microphone is required.
- After recording, the audio is converted to WAV (mono) and sent to the recognizer.
"""
)

# WebRTC client settings (optional)
CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
)

st.markdown("---")
st.markdown("#### 1) Start the recorder (click the play ▶️ button in the widget below). Speak for a few seconds.")
webrtc_ctx = webrtc_streamer(
    key="got-webrtc",
    mode=WebRtcMode.SENDONLY,
    client_settings=CLIENT_SETTINGS,
    audio_receiver_size=1024,
    video_frame_callback=None,
    audio_processor_factory=Recorder,
    async_processing=True,
)

st.markdown("#### 2) When you finish recording, click **Transcribe & Ask**.")
# Optional pause threshold slider for UX (not used by recognizer)
pause = st.slider("Pause threshold (UI only)", 0.5, 2.0, 0.8, 0.1)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Transcribe & Ask"):
        # check that webrtc is running and we've got an audio_processor
        if webrtc_ctx and webrtc_ctx.audio_processor:
            proc = webrtc_ctx.audio_processor
            frames = getattr(proc, "frames", None)
            sr_rate = getattr(proc, "sample_rate", None)

            if not frames or sr_rate is None:
                st.warning("No audio captured yet. Start the recorder and speak.")
            else:
                # Concatenate frames (each frame shape: channels x samples)
                try:
                    arr = np.concatenate(frames, axis=1)  # channels x total_samples
                except Exception as e:
                    st.error(f"Error concatenating audio frames: {e}")
                    arr = None

                if arr is not None:
                    # convert to mono: average channels if necessary
                    if arr.ndim == 2:
                        if arr.shape[0] > 1:
                            mono = np.mean(arr, axis=0)
                        else:
                            mono = arr[0]
                    else:
                        mono = arr

                    # normalize to float32 in range [-1.0, 1.0] if needed
                    # av returns int16-like types sometimes; soundfile will handle many dtypes but converting ensures consistency
                    mono = mono.astype(np.float32)

                    # write to in-memory WAV
                    bio = io.BytesIO()
                    sf.write(bio, mono.T, samplerate=sr_rate, format="WAV")
                    bio.seek(0)

                    # Transcribe with SpeechRecognition using AudioFile from memory
                    r = sr.Recognizer()
                    try:
                        with sr.AudioFile(bio) as source:
                            audio_data = r.record(source)
                            st.info("Transcribing audio...")
                            transcription = r.recognize_google(audio_data)
                    except sr.UnknownValueError:
                        transcription = "Sorry, I could not understand the audio."
                    except sr.RequestError:
                        transcription = "Speech recognition service unavailable."
                    except Exception as e:
                        transcription = f"Transcription error: {e}"

                    # Show user transcription and chatbot response
                    st.subheader("🗣 You said:")
                    st.write(transcription)
                    if transcription and not transcription.lower().startswith(("sorry", "transcription error")):
                        answer = chatbot(transcription)
                    else:
                        answer = transcription

                    st.subheader("🐺 Chatbot:")
                    st.write(answer)

                    # save to session history
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append(("You (speech)", transcription))
                    st.session_state.history.append(("Bot", answer))

                    # option to download the recorded WAV
                    st.download_button(
                        "📥 Download recorded WAV",
                        data=bio.getvalue(),
                        file_name=f"recording_{int(time.time())}.wav",
                        mime="audio/wav",
                    )
                # reset frames in processor for next recording
                proc.frames = []
                proc.sample_rate = None
        else:
            st.warning("Recorder not ready. Make sure you've allowed microphone permissions and started the recorder.")

with col2:
    if st.button("Clear chat history"):
        st.session_state.history = []

st.markdown("---")
st.subheader("Chat History")
if "history" in st.session_state and st.session_state.history:
    for role, msg in st.session_state.history:
        if role.startswith("You"):
            st.markdown(f"**{role}:** *{msg}*")
        else:
            st.markdown(f"**{role}:** {msg}")
else:
    st.info("No messages yet. Record audio and click 'Transcribe & Ask'.")

st.markdown("---")
st.markdown("If you run into problems on Streamlit Cloud, try in Chrome/Firefox desktop and ensure microphone permissions are granted.")