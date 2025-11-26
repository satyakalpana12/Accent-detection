Native Language Accent Detection

Using HuBERT + Logistic Regression

📌 1. Project Title

Native Language Accent Detection using HuBERT and Logistic Regression

🔗 2. Project Links

GitHub Repository:
👉 https://github.com/satyakalpana12/Accent-detection

Live Streamlit App:
👉 https://accent-detection-kalpana.streamlit.app

Google Drive (Code + Model + Checkpoints):
👉 https://drive.google.com/drive/folders/1mZ1P0SvNwg3FEPx5SFlB0hTiiEywfTmN?usp=sharing

📝 3. Project Description

This project automatically identifies a speaker’s native language accent from a .wav audio file.

The system uses:

HuBERT Base for audio feature extraction

Logistic Regression for classification

Streamlit Web App for easy online prediction

Users can upload audio or record through the microphone to get the predicted accent instantly.

📂 4. Repository Contents

File/Folder	Description

app.py	Main Streamlit application

models/	Contains trained model → accent_classifier_hubert.pkl

requirements.txt	All required Python libraries

README.md	Documentation of the entire project

📦 5. Required Packages & Versions

These are the main libraries used in development and deployment:

streamlit

librosa==0.11.0

soundfile==0.13.1

numpy

joblib

transformers

torch

torchaudio

matplotlib


(Already included in requirements.txt)

▶️ 6. How to Run the Project Locally

Step 1 — Download the Repository
git clone https://github.com/satyakalpana12/Accent-detection.git

Step 2 — Open the Project Folder
cd Accent-detection

Step 3 — Install Requirements
pip install -r requirements.txt

Step 4 — Run the Streamlit App
streamlit run app.py

Step 5 — Use the Application

Upload a .wav file OR

Record audio using your microphone

The system will show the predicted accent

🧠 7. How the Model Works

Audio is loaded and converted to mono

Resampled to 16,000 Hz

Normalized

HuBERT Base extracts a 768-dimensional embedding

The embedding is fed into Logistic Regression

Accent prediction is shown on the UI

🔍 8. Model Details
Component	Description

Feature Extractor	HuBERT Base (facebook/hubert-base-ls960)

Embedding Size	768

Classifier	Logistic Regression

Saved Model File	accent_classifier_hubert.pkl

Input Format	.wav 

✔️ 9. Final Notes

This project demonstrates how modern self-supervised learning models like HuBERT can be combined with classical ML techniques to perform accent classification effectively.
