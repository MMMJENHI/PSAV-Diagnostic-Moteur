import tensorflow as tf
import numpy as np

class VibrationAI:
    def __init__(self, model_path):
        # Chargement sécurisé du modèle .h5
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Erreur chargement modèle: {e}")

    def predict_event(self, segment):
        # Ajustement du format pour le modèle (Batch, Features, Channel)
        if len(segment) < 100: # Exemple de taille attendue
            segment = np.pad(segment, (0, 100 - len(segment)))
        
        segment = segment[np.newaxis, ..., np.newaxis]
        prediction = self.model.predict(segment, verbose=0)
        return prediction
