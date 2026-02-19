import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import scipy.fftpack

# --- 1. CONFIGURATION DE L'INTERFACE ---
st.set_page_config(page_title="IA Diagnostic Vibratoire", layout="wide")
st.title("🚜 Système Expert de Maintenance Prédictive")
st.markdown("Analyse des signaux vibratoires par Deep Learning")

# --- 2. CHARGEMENT DES RESSOURCES (Modèle IA) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "expert_vibration_tensorflow.h5")

@st.cache_resource
def load_ai_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_ai_model()

# --- 3. GESTION DES DONNÉES (Sidebar) ---
with st.sidebar:
    st.header("📂 Sélection des données")
    etat = st.selectbox("État réel du moteur :", ["sain", "defectueux"])
    
    data_folder = os.path.join(BASE_DIR, "data", etat)
    if os.path.exists(data_folder):
        fichiers = [f for f in os.listdir(data_folder) if f.endswith('.wav')]
        selected_file = st.selectbox("Choisir un échantillon :", fichiers)
        path = os.path.join(data_folder, selected_file)
        
        # Chargement du signal audio
        sig, sr = librosa.load(path, sr=None)
        df = pd.DataFrame(sig, columns=["Amplitude"])

# --- 4. ANALYSE ET VISUALISATION ---
if 'sig' in locals():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Signal Temporel (Brut)")
        st.line_chart(df.iloc[:2000], height=300)
        st.audio(path) # Permet d'écouter le bruit du moteur

    with col2:
        st.subheader("📊 Analyse Fréquentielle (FFT)")
        N = len(sig)
        yf = scipy.fftpack.fft(sig)
        xf = np.linspace(0.0, 1.0/(2.0/sr), N//2)
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='orangered')
        ax.set_xlabel("Fréquence (Hz)")
        st.pyplot(fig)

    # --- 5. DIAGNOSTIC IA (Le Verdict) ---
    st.divider()
    st.subheader("🧠 Verdict du Système Expert")
    
    if model:
        # Extraction des 2 caractéristiques attendues par ton modèle Dense
        # On utilise le RMS (énergie) et le Pic Maximum
        rms = np.sqrt(np.mean(sig**2))
        peak = np.max(np.abs(sig))
        
        # Préparation du vecteur pour l'IA (Shape: 1, 2)
        input_data = np.array([[rms, peak]], dtype=np.float32)
        
        # Prédiction
        prediction = model.predict(input_data, verbose=0)
        score = float(prediction[0][0])
        
        # Affichage stylisé du résultat
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if score > 0.5:
                st.error(f"### 🚨 ALERTE : ANOMALIE DÉTECTÉE")
                st.write("**Diagnostic :** Signature vibratoire caractéristique d'une défaillance.")
            else:
                st.success(f"### ✅ ÉTAT : MOTEUR CONFORME")
                st.write("**Diagnostic :** Le signal correspond à un fonctionnement normal.")
        
        with res_col2:
            st.metric("Indice de Risque", f"{score:.2%}")
            # Petit rappel technique pour le jury
            st.caption(f"Features extraites : RMS={rms:.4f}, Peak={peak:.4f}")
