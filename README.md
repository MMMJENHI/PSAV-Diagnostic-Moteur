# 🚜 PSAV : Plateforme de Surveillance et d'Analyse Vibratoire

![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

## 📋 Présentation du Projet
Ce projet est un **Système Expert de Maintenance Prédictive**. Il permet d'analyser l'état de santé de moteurs industriels à partir de signaux vibratoires. L'application utilise l'Intelligence Artificielle pour distinguer un moteur sain d'un moteur présentant une défaillance (roulement, déséquilibre, etc.).

## 🚀 Fonctionnalités
* **Visualisation temporelle** : Affichage de l'amplitude du signal brut.
* **Analyse fréquentielle (FFT)** : Transformation du signal pour identifier les fréquences de défaut.
* **Diagnostic IA** : Classification automatique via un modèle Deep Learning (TensorFlow/Keras).
* **Indicateurs clés** : Calcul du RMS (énergie) et du Peak (chocs) pour l'aide à la décision.

## 🛠️ Architecture Technique
L'application suit un pipeline de données rigoureux :
1. **Acquisition** : Lecture de fichiers `.wav` (simulant un accéléromètre).
2. **Prétraitement** : Nettoyage et normalisation avec `Librosa`.
3. **Extraction** : Calcul des descripteurs statistiques.
4. **Verdict** : Inférence via le modèle `expert_vibration_tensorflow.h5`.



## 📁 Structure du dépôt
* `app.py` : Code principal de l'interface Streamlit.
* `models/` : Contient le modèle d'IA entraîné.
* `data/` : Échantillons de signaux (Sains et Défectueux).
* `requirements.txt` : Liste des bibliothèques nécessaires.

## 👷 Auteur
* **MMMJENHI** - *Développement et Intégration IA*
