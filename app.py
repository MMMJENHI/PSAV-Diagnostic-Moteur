import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf

# --- 1. CONFIGURATION ET INITIALISATION ---
st.set_page_config(page_title="Diagnostic Vibratoire", layout="wide")
st.title("🚜 Système Expert : Diagnostic Moteur")

# On définit les variables pour éviter les "NameError"
df = None
model = None

# --- 2. GESTION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "expert_vibration_tensorflow.h5")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- 3. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_my_model()

# Barre latérale
st.sidebar.header("⚙️ Paramètres")
if model:
    st.sidebar.success("✅ Modèle IA chargé")
else:
    st.sidebar.error("❌ Modèle IA introuvable")

# --- 4. CHARGEMENT DES DONNÉES ---
st.sidebar.subheader("Sélection des données")
source = st.sidebar.radio("Source :", ["Exemples du projet", "Télécharger un CSV"])

if source == "Exemples du projet":
    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if files:
            selected = st.sidebar.selectbox("Fichier :", files)
            df = pd.read_csv(os.path.join(DATA_DIR, selected))
else:
    uploaded = st.sidebar.file_uploader("Fichier CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)

# --- 5. AFFICHAGE ET ANALYSE ---
if df is not None:
    st.subheader("📊 Visualisation du Signal")
    # On affiche les 1000 premières lignes de la 1ère colonne
    st.line_chart(df.iloc[:1000, 0])
    
    if model:
        st.subheader("🧠 Résultat du Diagnostic")
        # On prépare la donnée pour le modèle
        try:
            sample = df.iloc[:1000, 0].values.reshape(1, 1000, 1)
            prediction = model.predict(sample)
            score = prediction[0][0]
            
            if score > 0.5:
                st.error(f"🚨 ANOMALIE DÉTECTÉE (Probabilité : {score:.2%})")
            else:
                st.success(f"✅ MOTEUR SAIN (Confiance : {1-score:.2%})")
        except Exception as e:
            st.info("Signal chargé. Prêt pour l'analyse manuelle.")
else:
    st.info("👋 Bienvenue ! Veuillez choisir un fichier CSV dans le menu à gauche pour commencer l'analyse.")
