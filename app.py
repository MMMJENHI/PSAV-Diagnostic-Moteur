import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="IA Diagnostic Vibratoire", layout="wide")
st.title("🚜 Système Expert : Diagnostic Moteur (WAV)")

# --- INITIALISATION ---
df = None
model = None

# Gestion des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "expert_vibration_tensorflow.h5")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- 1. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except:
            return None
    return None

model = load_my_model()

# Sidebar
st.sidebar.header("⚙️ Contrôle")
if model:
    st.sidebar.success("✅ IA : Modèle chargé")
else:
    st.sidebar.error("❌ IA : Modèle non trouvé")

# --- 2. NAVIGATION DANS LES DOSSIERS (SAIN / DEFECTUEUX) ---
st.sidebar.subheader("Explorateur de données")
source = st.sidebar.radio("Source :", ["Exemples du projet", "Télécharger WAV"])

if source == "Exemples du projet":
    # On propose les deux sous-dossiers que tu as sur ton PC
    categorie = st.sidebar.selectbox("État du moteur :", ["sain", "defectueux"])
    dossier_cible = os.path.join(DATA_DIR, categorie)
    
    if os.path.exists(dossier_cible):
        fichiers = [f for f in os.listdir(dossier_cible) if f.endswith('.wav')]
        if fichiers:
            nom_fichier = st.sidebar.selectbox("Choisir un échantillon :", fichiers)
            chemin_complet = os.path.join(dossier_cible, nom_fichier)
            
            # Lecture du fichier WAV avec Librosa
            signal, sr = librosa.load(chemin_complet, sr=None)
            df = pd.DataFrame(signal, columns=["Amplitude"])
        else:
            st.sidebar.warning("Aucun fichier .wav trouvé.")
    else:
        st.sidebar.error("Dossier data/ non trouvé sur le serveur.")

# --- 3. AFFICHAGE DES RÉSULTATS ---
if df is not None:
    st.subheader(f"📊 Signal Temporel - {categorie if 'categorie' in locals() else 'Upload'}")
    
    # Graphique du signal
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.iloc[:2000], color='#0077b6')
    ax.set_title("Onde vibratoire (Zoom sur les 2000 premiers points)")
    st.pyplot(fig)

    # Diagnostic IA
    if model:
        st.divider()
        st.subheader("🧠 Diagnostic de l'Intelligence Artificielle")
        try:
            # On prépare la donnée pour ton modèle (ex: 1000 points)
            input_ia = df.iloc[:1000, 0].values.reshape(1, 1000, 1)
            pred = model.predict(input_ia)
            score = pred[0][0]

            if score > 0.5:
                st.error(f"🚨 ANOMALIE DÉTECTÉE (Score : {score:.2%})")
            else:
                st.success(f"✅ MOTEUR SAIN (Confiance : {1-score:.2%})")
        except Exception as e:
            st.info("Signal chargé. Prêt pour l'analyse visuelle.")
else:
    st.info("👈 Sélectionnez un fichier .wav dans le dossier 'sain' ou 'defectueux' pour commencer.")
