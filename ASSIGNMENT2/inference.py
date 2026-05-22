"""
Inferenza — classifica una singola immagine con il modello finale.

Uso:
    python inference.py <path_immagine>

Carica vocabulary_final.pkl e classifier_final.pkl, estrae i descrittori
con lo stesso tipo usato in training (letto da cv_results.csv), costruisce
l'istogramma BoW e predice la classe.

Requisiti: la pipeline completa (00→05) deve essere già stata eseguita.
    - models/classifier_final.pkl  (prodotto da 05_train_final.py)
    - models/vocabulary_final.pkl  (prodotto da 05_train_final.py)
    - results/cv_results.csv       (prodotto da 04_cross_validation.py)
"""

import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize

from config import FINAL_MODEL_FILE, FINAL_VOCAB_FILE, CV_RESULTS_FILE
from utils import load_pickle, load_image_gray


def get_descriptor_type() -> str:
    """
    Legge da cv_results.csv quale tipo di descrittore ha vinto la CV.

    Non hardcodiamo il tipo di descrittore: lo leggiamo dal CSV così che
    se si ri-esegue la pipeline con ORB abilitato, inference si adatta
    automaticamente senza richiedere modifiche al codice.

    Ritorna
    -------
    str : "SIFT" o "ORB" (il descrittore della riga con f1_mean più alto)
    """
    if not CV_RESULTS_FILE.exists():
        print("[ERR] cv_results.csv non trovato. Esegui prima la pipeline completa.")
        sys.exit(1)
    df   = pd.read_csv(CV_RESULTS_FILE)
    best = df.loc[df["f1_mean"].idxmax()]
    return best["descriptor"]


def make_extractor(desc_type: str):
    """Restituisce il rilevatore OpenCV corretto. Identica alle fasi precedenti."""
    if desc_type == "SIFT":
        return cv2.SIFT_create()
    if desc_type == "ORB":
        return cv2.ORB_create(nfeatures=500)
    raise ValueError(f"Tipo descrittore non supportato: {desc_type}")


def extract_bow(img: np.ndarray, extractor, kmeans) -> np.ndarray:
    """
    Estrae l'istogramma BoW L2-normalizzato da un'immagine.

    Replica esattamente il preprocessing usato in 03_compute_bow.py:
    stessa estrazione descrittori → stesso hard assignment → stessa
    normalizzazione L2. È fondamentale che inference usi la stessa
    pipeline di training: qualsiasi differenza (es. normalizzazione
    diversa) invaliderebbe la predizione del classificatore.

    Parametri
    ---------
    img       : immagine in scala di grigi
    extractor : oggetto SIFT o ORB già inizializzato
    kmeans    : modello MiniBatchKMeans (vocabulary_final.pkl)

    Ritorna
    -------
    np.ndarray di shape (1, K) float32, L2-normalizzato
        Shape 2D perché clf.predict si aspetta (n_samples, n_features).
    """
    k = kmeans.n_clusters
    _, descs = extractor.detectAndCompute(img, None)

    if descs is None or len(descs) == 0:
        print("[warn] Nessun keypoint trovato nell'immagine.")
        # Restituisce un vettore zero di shape (1, K):
        # il classificatore produrrà una predizione, ma sarà poco affidabile
        # perché non c'è nessuna informazione visiva estratta dall'immagine.
        return np.zeros((1, k), dtype=np.float32)

    words = kmeans.predict(descs.astype(np.float32))

    # reshape(1, -1) trasforma l'istogramma da (K,) a (1, K):
    # sklearn richiede che l'input a predict sia sempre 2D (n_samples, n_features).
    # Per una singola immagine n_samples=1, quindi serve la dimensione batch esplicita.
    hist  = np.bincount(words, minlength=k).astype(np.float32).reshape(1, -1)
    hist  = normalize(hist, norm="l2")
    return hist


def main():
    if len(sys.argv) != 2:
        print("Uso: python inference.py <path_immagine>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"[ERR] File non trovato: {img_path}")
        sys.exit(1)

    # ── Verifica artefatti obbligatori ────────────────────────────────────────
    # Controllo esplicito prima del caricamento per produrre messaggi d'errore
    # chiari invece di un generico FileNotFoundError da load_pickle.
    for path, name in [(FINAL_MODEL_FILE, "classifier_final.pkl"),
                       (FINAL_VOCAB_FILE,  "vocabulary_final.pkl")]:
        if not path.exists():
            print(f"[ERR] {name} non trovato. Esegui prima 05_train_final.py")
            sys.exit(1)

    clf       = load_pickle(FINAL_MODEL_FILE)
    kmeans    = load_pickle(FINAL_VOCAB_FILE)
    desc_type = get_descriptor_type()

    print(f"Modello caricato  : {FINAL_MODEL_FILE.name}")
    print(f"Vocabolario       : {FINAL_VOCAB_FILE.name}  (K={kmeans.n_clusters})")
    print(f"Descrittore       : {desc_type}")
    print(f"Immagine          : {img_path}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    # Conversione in scala di grigi: SIFT e ORB operano su immagini single-channel.
    # L'informazione cromatica viene ignorata, coerentemente con il training.
    img = load_image_gray(img_path)
    if img is None:
        print(f"[ERR] Impossibile leggere l'immagine: {img_path}")
        sys.exit(1)

    # ── Estrazione feature e predizione ───────────────────────────────────────
    extractor = make_extractor(desc_type)
    bow       = extract_bow(img, extractor, kmeans)

    # clf.predict restituisce un array anche per input singolo: [0] estrae la stringa.
    pred = clf.predict(bow)[0]
    print(f"\nClasse predetta   : {pred}")

    # predict_proba è disponibile solo se il classificatore è stato addestrato
    # con probability=True (SVC di default non lo è) oppure se supporta nativamente
    # le probabilità (RandomForest). hasattr evita un AttributeError silenzioso.
    # Se disponibile, mostra le top-3 classi con la relativa probabilità stimata:
    # utile per capire se la predizione è netta (es. 90%) o incerta (es. 40%).
    if hasattr(clf, "predict_proba"):
        probs      = clf.predict_proba(bow)[0]
        classes    = clf.classes_
        top3_idx   = np.argsort(probs)[::-1][:3]  # indici delle 3 prob più alte
        print("Top-3 classi:")
        for i in top3_idx:
            print(f"  {classes[i]:<25} {probs[i]*100:.1f}%")


if __name__ == "__main__":
    main()
