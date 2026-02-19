# --- APRÈS LE CHARGEMENT DU MODÈLE ---

if df is not None:
    st.divider()
    st.subheader("📊 Analyse du Signal Vibratoire")
    
    # 1. Affichage du graphique
    # On suppose que la première colonne contient les vibrations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.line_chart(df.iloc[:1000, 0]) # Affiche les 1000 premiers points
        st.caption("Signal temporel des vibrations (accéléromètre)")

    with col2:
        st.write("🔍 **Statistiques du signal :**")
        st.write(f"Moyenne : {df.iloc[:,0].mean():.4f}")
        st.write(f"Max (Crête) : {df.iloc[:,0].max():.4f}")

    # 2. Lancement du Diagnostic par l'IA
    st.divider()
    st.subheader("🧠 Verdict du Système Expert")
    
    # On prépare la donnée pour le modèle (souvent un tableau de 1000 points)
    try:
        # On redimensionne pour correspondre à l'entrée du réseau de neurones
        input_data = df.iloc[:1000, 0].values.reshape(1, 1000, 1)
        
        # L'IA fait sa prédiction
        prediction = model.predict(input_data)
        probabilite = prediction[0][0]

        if probabilite > 0.5:
            st.error(f"🚨 ALERTE : ANOMALIE DÉTECTÉE ({probabilite:.2%})")
            st.info("💡 **Diagnostic :** Usure probable des roulements ou balourd détecté.")
        else:
            st.success(f"✅ ÉTAT NORMAL ({1 - probabilite:.2%})")
            st.info("💡 **Diagnostic :** Le moteur fonctionne dans les plages de tolérance.")
            
    except Exception as e:
        st.warning("⚠️ Format de données : Assurez-vous que le CSV contient au moins 1000 lignes.")
else:
    # Ce message s'affiche tant qu'aucun fichier n'est choisi
    st.info("👈 Veuillez sélectionner un fichier dans la barre latérale pour lancer l'analyse.")
