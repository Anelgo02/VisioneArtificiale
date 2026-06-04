# inference.py
# Dato il path di una immagine RGB, predice la classe con il miglior modello.
# Requisiti del task:
#   1. Carica l'immagine
#   2. Pre-processa con lo STESSO preprocessing del training (OpenCV)
#   3. Predice e stampa il risultato

import sys
import cv2
import numpy as np
import tensorflow as tf

import config


def preprocess_for_inference(image_path):
    """
    Carica e pre-processa un'immagine esattamente come in training.
    CRITICO: se il preprocessing in inferenza è diverso da quello in training,
    le performance degradano (train/test mismatch).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Impossibile caricare: {image_path}")

    # BGR -> RGB (OpenCV legge in BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize identico al training
    img = cv2.resize(img, (config.IMG_SIZE[1], config.IMG_SIZE[0]))

    # Normalizzazione [0, 1] identica al training
    img = img.astype(np.float32) / 255.0

    # Aggiungi dimensione batch: (H, W, C) -> (1, H, W, C)
    # Necessario perché il modello si aspetta un batch in input.
    img = np.expand_dims(img, axis=0)
    return img


def predict(image_path, model_path=config.BEST_MODEL_PATH):
    """
    Predice la classe di una singola immagine.

    Parametri
    ----------
    image_path : str - path all'immagine RGB
    model_path : str - path al modello .keras salvato
    """
    # 1. Carica il modello
    print(f"Caricamento modello: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 2. Pre-processa l'immagine
    img = preprocess_for_inference(image_path)

    # 3. Predici
    probs = model.predict(img, verbose=0)[0]   # shape: (21,)

    # Classe con probabilità massima
    predicted_idx  = np.argmax(probs)
    predicted_label = config.UC_CLASS_NAMES[predicted_idx]
    confidence      = probs[predicted_idx]

    # Stampa risultato
    print(f"\nImmagine:      {image_path}")
    print(f"Classe predetta: {predicted_label}")
    print(f"Confidenza:      {confidence:.2%}")

    # Top-5 classi per debug/analisi
    top5_idx  = np.argsort(probs)[::-1][:5]
    print("\nTop-5 classi:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  {rank}. {config.UC_CLASS_NAMES[idx]:<25} {probs[idx]:.4f}")

    return predicted_label, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python inference.py <path_immagine> [path_modello]")
        sys.exit(1)

    img_path = sys.argv[1]
    mdl_path = sys.argv[2] if len(sys.argv) > 2 else config.BEST_MODEL_PATH

    predict(img_path, mdl_path)
