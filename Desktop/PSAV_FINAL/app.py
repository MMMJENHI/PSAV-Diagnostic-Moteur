import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from src.psav_core import PSAVEngine
from src.ai_classifier import VibrationAI

st.set_page_config(page_title="PSAV Vibration Expert", layout="wide")

st.title("🛡️ Système Expert de Surveillance Vibratoire")
st.markdown("---")

# Chargement de l'IA
@st.cache_resource
def load_model():
    return VibrationAI('models/expert_vibration_tensorflow..h5')

ai = load_model()

# Sidebar pour les réglages
st.sidebar.header("Réglages Algorithme")
sensibilite = st.sidebar.slider("Facteur de Seuil (Sensibilité)", 1.0, 10.0, 3.5)
upload = st.file_uploader("Importer un signal (.wav)", type=["wav"])

if upload:
    sr, signal = wavfile.read(upload)
    if signal.ndim > 1: signal = signal[:, 0]
    signal = signal / (np.max(np.abs(signal)) + 1e-9)

    # Exécution PSAV
    engine = PSAVEngine(sr, factor=sensibilite)
    peaks, thresholds = engine.analyze(signal)

    # Affichage
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, label="Vibration brute", color="silver")
    ax.plot(thresholds, label="Seuil Adaptatif", color="red", linestyle="--")
    ax.scatter(peaks, signal[peaks], color="blue", label="Pics détectés")
    ax.legend()
    st.pyplot(fig)

    st.subheader(f"🔍 Analyse de {len(peaks)} événements")
    
    if len(peaks) > 0:
        # Analyse du premier pic avec l'IA
        seg = signal[peaks[0]-50 : peaks[0]+50]
        res = ai.predict_event(seg)
        st.info(f"Probabilité d'anomalie pour le premier pic : {res[0][0]*100:.2f}%")
