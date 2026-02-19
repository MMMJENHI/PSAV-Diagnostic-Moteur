import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import scipy.fftpack

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="IA Diagnostic Moteur", layout="wide")

# TITRE STYLE EXAMEN
st.title("🚜 Système Expert de Diagnostic Vibratoire")
st.markdown("---")

# INITIALISATION DES VARIABLES
df = None
model = None

# CHEMINS DES DOSSIERS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "expert_vibration_tensorflow.h5")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- 1. CHARGEMENT DU MODÈLE IA ---
@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_my_model()

# BARRE LATÉRALE (SIDEBAR)
with st.sidebar:
    st.header("⚙️ Contrôle & État")
    if model:
        st.success("✅ IA : Modèle chargé")
    else:
        st.error("❌ IA : Modèle introuvable")
    
    st.divider()
    
    st.subheader("Source des données")
    source = st.radio("Choisir la source :", ["Exemples du projet", "Télécharger un WAV"])

    if source == "Exemples du projet":
        cat = st.selectbox("État du moteur :", ["sain", "defectueux"])
        folder = os.path.join(DATA_DIR, cat)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.wav')]
            selected = st.selectbox("Choisir un échantillon :", files)
            path = os.path.join(folder, selected)
            # Chargement audio
            sig, sr = librosa.load(path, sr=None)
            df = pd.DataFrame(sig, columns=["Amplitude"])

# --- 2. AFFICHAGE ET VERDICT ---
if df is not None:
    # Ligne 1 : Les graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Signal Temporel")
        st.line_chart(df.iloc[:2000], height=250)
        st.audio(path) # Pour écouter le moteur !

    with col2:
        st.subheader("📊 Analyse Fréquentielle (FFT)")
        N = len(df)
        yf = scipy.fftpack.fft(df.iloc[:, 0].values)
        xf = np.linspace(0.0, 1.0/(2.0/sr), N//2)
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='red')
        st.pyplot(fig)

    # Ligne 2 : Le Verdict IA
    st.divider()
    st.subheader("🧠 Résultat du Diagnostic Automatique")
    
    if model:
        # On prépare 1000 points pour l'IA
        input_data = df.iloc[:1000, 0].values.reshape(1, 1000, 1)
        pred = model.predict(input_data)
        score = pred[0][0]

        if score > 0.5:
            st.error(f"### 🚨 VERDICT : DÉFAUT DÉTECTÉ")
            st.progress(float(score))
            st.write(f"Probabilité de panne : **{score:.2%}**")
        else:
            st.success(f"### ✅ VERDICT : MOTEUR SAIN")
            st.progress(float(1-score))
            st.write(f"Confiance : **{1-score:.2%}**")
else:
    st.info("👋 Veuillez sélectionner un fichier dans la barre latérale pour lancer l'analyse.")
