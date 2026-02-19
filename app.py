import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Diagnostic Moteur", layout="wide")

# --- FONCTION DE RECHERCHE AUTOMATIQUE ---
def trouver_chemin(nom_dossier):
    # On cherche le dossier dans le répertoire courant et les sous-répertoires
    for root, dirs, files in os.walk("."):
        if nom_dossier in dirs:
            return os.path.join(root, nom_dossier)
    return None

# Localisation des dossiers
chemin_data = trouver_chemin("data")
chemin_models = trouver_chemin("models")

# --- BARRE LATÉRALE ---
st.sidebar.title("🔍 État du Système")

# 1. Chargement du Modèle
model_file = None
if chemin_models:
    model_file = os.path.join(chemin_models, "vibration_model.h5")
    if os.path.exists(model_file):
        try:
            @st.cache_resource
            def load_model():
                return tf.keras.models.load_model(model_file)
            model = load_model()
            st.sidebar.success("✅ IA : Modèle chargé")
        except Exception as e:
            st.sidebar.error("Erreur de lecture du .h5")
    else:
        st.sidebar.warning("⚠️ Fichier .h5 introuvable dans /models")
else:
    st.sidebar.error("❌ Dossier /models introuvable")

# 2. Gestion des données
st.sidebar.divider()
source = st.sidebar.radio("Source :", ("Mon Ordinateur", "GitHub (Dossier Data)"))

file_to_process = None

if source == "Mon Ordinateur":
    file_to_process = st.file_uploader("Charger un CSV", type=["csv"])
else:
    if chemin_data:
        fichiers = [f for f in os.listdir(chemin_data) if f.endswith('.csv')]
        if fichiers:
            choix = st.sidebar.selectbox("Choisir un exemple :", fichiers)
            file_to_process = os.path.join(chemin_data, choix)
            st.sidebar.success(f"Fichier trouvé : {choix}")
        else:
            st.sidebar.error("Le dossier 'data' est vide sur GitHub.")
    else:
        st.sidebar.error("❌ Dossier 'data' introuvable")

# --- CORPS DE L'APPLICATION ---
st.title("🚀 Diagnostic Vibratoire Moteur")

if file_to_process:
    try:
        df = pd.read_csv(file_to_process)
        signal = df.iloc[:, 0].values
        
        st.subheader("Analyse du Signal")
        st.line_chart(signal[:1000])
        
        # Ici tu peux ajouter tes prédictions model.predict(signal)
        st.success("Analyse terminée avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
else:
    st.info("Sélectionnez une donnée pour démarrer l'analyse.")
    # Debug pour t'aider : affiche ce que le serveur voit réellement
    with st.expander("DEBUG : Structure du serveur"):
        st.write("Dossiers présents :", os.listdir("."))
