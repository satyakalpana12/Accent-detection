import streamlit as st
import numpy as np
import joblib
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel

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
st.write("Upload a **.wav** audio file for prediction.")

uploaded_file = st.file_uploader("Upload your audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Load audio
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    y, sr = librosa.load(audio_path, sr=None)

    # Preprocess → Extract embeddings → Predict
    y = preprocess_audio(y, sr)
    emb = extract_hubert(y)
    pred = clf.predict([emb])[0]

    st.success(f"### 🟢 Predicted Accent: **{pred}**")

st.write("---")
st.caption("Model: HuBERT Base + Logistic Regression")
