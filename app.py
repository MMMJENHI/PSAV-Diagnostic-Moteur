import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import scipy.fftpack

# --- CONFIGURATION ---
st.set_page_config(page_title="Diagnostic IA", layout="wide")
st.title("🚜 Système Expert Vibratoire")

# --- CHARGEMENT DU MODÈLE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "expert_vibration_tensorflow.h5")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_model()

# --- SIDEBAR / SÉLECTION ---
st.sidebar.header("📁 Données")
categorie = st.sidebar.selectbox("État :", ["sain", "defectueux"])
data_path = os.path.join(BASE_DIR, "data", categorie)

if os.path.exists(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    selected_file = st.sidebar.selectbox("Fichier :", files)
    file_path = os.path.join(data_path, selected_file)
    
    # Lecture du signal
    sig, sr = librosa.load(file_path, sr=None)
    df = pd.DataFrame(sig, columns=["Amplitude"])

    # --- AFFICHAGE ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Signal Temporel")
        st.line_chart(df.iloc[:2000])
        st.audio(file_path)

    with col2:
        st.subheader("📊 Spectre (FFT)")
        N = len(sig)
        yf = scipy.fftpack.fft(sig)
        xf = np.linspace(0.0, 1.0/(2.0/sr), N//2)
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='red')
        st.pyplot(fig)

    # --- VERDICT IA (CORRECTION VALUEERROR) ---
    st.divider()
    st.subheader("🧠 Verdict de l'IA")
    
    if model:
        # ON EXTRAIT 2 CARACTÉRISTIQUES (car le modèle attend shape=2)
        mean_val = np.mean(np.abs(sig))
        std_val = np.std(sig)
        
        # Formatage précis pour éviter l'erreur axis -1
        input_for_ai = np.array([[mean_val, std_val]], dtype=np.float32)
        
        prediction = model.predict(input_for_ai, verbose=0)
        score = float(prediction[0][0])
        
        if score > 0.5:
            st.error(f"### 🚨 DÉFAUT DÉTECTÉ (Score: {score:.2f})")
        else:
            st.success(f"### ✅ MOTEUR SAIN (Confiance: {1-score:.2%})")
else:
    st.error("Dossier data introuvable.")
