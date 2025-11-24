import streamlit as st
import numpy as np
import joblib
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from audiorecorder import audiorecorder

# -------------------------------------------------
# Load model
# -------------------------------------------------
MODEL_PATH = "models/accent_classifier_hubert.pkl"
clf = joblib.load(MODEL_PATH)

# -------------------------------------------------
# Load HuBERT feature extractor + model
# -------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-base-ls960"
)
hubert = HubertModel.from_pretrained(
    "facebook/hubert-base-ls960"
).to(device)
hubert.eval()

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def preprocess_audio(y, sr):
    """Convert to mono, resample to 16000 Hz, normalize"""
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return librosa.util.normalize(y)


def extract_hubert(y):
    """Extract HuBERT embeddings"""
    inputs = feature_extractor(
        y,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = hubert(**inputs)

    emb = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("🎤 Native Language Accent Detection")
st.write("Upload a **.wav** audio file OR record using microphone.")

st.subheader("Choose Input Method")

# -----------------------------
# Upload option
# -----------------------------
uploaded_file = st.file_uploader("Upload your audio (.wav)", type=["wav"])

# -----------------------------
# Microphone option
# -----------------------------
st.write("Or record using microphone:")
audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")

audio_data = None

# If microphone audio exists
if len(audio) > 0:
    st.audio(audio.tobytes())
    audio_data = audio.tobytes()

# If file uploader used
if uploaded_file is not None:
    st.audio(uploaded_file)
    audio_data = uploaded_file.read()

# -----------------------------
# If audio is provided → predict
# -----------------------------
if audio_data is not None:

    # Load audio from bytes
    try:
        y, sr = librosa.load(
            librosa.util.buf_to_float(audio_data),
            sr=None
        )
    except:
        # fallback for uploaded_file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data)
            path = tmp.name
        y, sr = librosa.load(path, sr=None)

    y = preprocess_audio(y, sr)
    emb = extract_hubert(y)

    pred = clf.predict([emb])[0]

    st.success(f"### 🟢 Predicted Accent: **{pred}**")

st.write("---")
st.caption("Model: HuBERT Base + Logistic Regression")
