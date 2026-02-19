import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf

# 1. On définit où se trouve le dossier du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. On crée le chemin vers ton fichier .h5 (le nom exact que nous avons trouvé)
CHEMIN_MODELE = os.path.join(BASE_DIR, "models", "expert_vibration_tensorflow.h5")

# 3. On demande à Python de charger le cerveau de l'IA
if os.path.exists(CHEMIN_MODELE):
    try:
        # Cette fonction charge le modèle une seule fois pour ne pas ralentir l'app
        @st.cache_resource
        def load_my_model():
            return tf.keras.models.load_model(CHEMIN_MODELE)
        
        model = load_my_model()
        st.sidebar.success("✅ IA : Modèle chargé avec succès")
    except Exception as e:
        st.sidebar.error(f"Erreur technique lors du chargement : {e}")
else:
    # Si le chemin est faux, ce message s'affichera
    st.sidebar.warning(f"⚠️ Fichier introuvable à l'adresse : {CHEMIN_MODELE}")
