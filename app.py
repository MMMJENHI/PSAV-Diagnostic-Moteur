# --- ETAPE 2 : Source des données ---
st.sidebar.title("Source des donnees")
source = st.sidebar.radio("Choisir la source :", ("Mon Ordinateur", "Exemples du projet (GitHub)"))

file_to_process = None
data_path_on_disk = None

if source == "Mon Ordinateur":
    uploaded_file = st.file_uploader("Charger un fichier", type=["wav", "csv"])
    if uploaded_file is not None:
        file_to_process = uploaded_file
else:
    # On définit le dossier (vérifie bien s'il faut 'data' ou 'Data')
    target_dir = "data" 
    
    if os.path.exists(target_dir):
        test_files = [f for f in os.listdir(target_dir) if f.endswith(('.wav', '.csv'))]
        if test_files:
            selected_test = st.selectbox("Choisir un fichier de test :", test_files)
            data_path_on_disk = os.path.join(target_dir, selected_test)
            file_to_process = data_path_on_disk
        else:
            st.error(f"Le dossier '{target_dir}' est vide sur GitHub.")
    else:
        st.error(f"Dossier '{target_dir}' introuvable sur GitHub.")

# --- ETAPE 3 : Traitement ---
if file_to_process is not None:
    st.divider()
    
    # Lecture des données réelles (CSV ou WAV)
    try:
        if source == "Mon Ordinateur":
            # Si c'est un upload
            df = pd.read_csv(file_to_process)
        else:
            # Si c'est un fichier du dossier data
            df = pd.read_csv(data_path_on_disk)
        
        # On extrait le signal (on suppose que c'est la 1ère colonne)
        signal = df.iloc[:, 0].values
        st.success("✅ Données chargées avec succès depuis le fichier.")
    except Exception as e:
        st.warning("Lecture directe impossible, utilisation d'un signal simulé.")
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 50 * t) + 0.3 * np.random.randn(1000)

    # Affichage des graphiques (le reste de ton code ne change pas)
    st.subheader("1. Signal Vibratoire (Temporel)")
    st.line_chart(signal)
    # ... (le reste de ta FFT ici)
