"""
Inferenza — classifica una singola immagine con il modello finale.

Uso:
    python inference.py <path_immagine>

Carica vocabulary_final.pkl e classifier_final.pkl, estrae i descrittori
con lo stesso tipo usato in training (letto da cv_results.csv), costruisce
l'istogramma BoW e predice la classe.
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
    """Legge da cv_results.csv quale descrittore ha vinto."""
    if not CV_RESULTS_FILE.exists():
        print("[ERR] cv_results.csv non trovato. Esegui prima la pipeline completa.")
        sys.exit(1)
    df   = pd.read_csv(CV_RESULTS_FILE)
    best = df.loc[df["f1_mean"].idxmax()]
    return best["descriptor"]


def make_extractor(desc_type: str):
    if desc_type == "SIFT":
        return cv2.SIFT_create()
    if desc_type == "ORB":
        return cv2.ORB_create(nfeatures=500)
    raise ValueError(f"Tipo descrittore non supportato: {desc_type}")


def extract_bow(img: np.ndarray, extractor, kmeans) -> np.ndarray:
    """Estrae l'istogramma BoW L2-normalizzato da un'immagine."""
    k = kmeans.n_clusters
    _, descs = extractor.detectAndCompute(img, None)

    if descs is None or len(descs) == 0:
        print("[warn] Nessun keypoint trovato nell'immagine.")
        return np.zeros((1, k), dtype=np.float32)

    words = kmeans.predict(descs.astype(np.float32))
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

    # ── Caricamento modello ───────────────────────────────────────────────────
    for path, name in [(FINAL_MODEL_FILE, "classifier_final.pkl"),
                       (FINAL_VOCAB_FILE,  "vocabulary_final.pkl")]:
        if not path.exists():
            print(f"[ERR] {name} non trovato. Esegui prima 05_train_final.py")
            sys.exit(1)

    clf    = load_pickle(FINAL_MODEL_FILE)
    kmeans = load_pickle(FINAL_VOCAB_FILE)
    desc_type = get_descriptor_type()

    print(f"Modello caricato  : {FINAL_MODEL_FILE.name}")
    print(f"Vocabolario       : {FINAL_VOCAB_FILE.name}  (K={kmeans.n_clusters})")
    print(f"Descrittore       : {desc_type}")
    print(f"Immagine          : {img_path}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    img = load_image_gray(img_path)
    if img is None:
        print(f"[ERR] Impossibile leggere l'immagine: {img_path}")
        sys.exit(1)

    # ── Estrazione feature e predizione ───────────────────────────────────────
    extractor = make_extractor(desc_type)
    bow       = extract_bow(img, extractor, kmeans)
    pred      = clf.predict(bow)[0]

    # SVC con probability=False non ha predict_proba; se abilitato mostriamo top-3
    print(f"\nClasse predetta   : {pred}")

    if hasattr(clf, "predict_proba"):
        probs      = clf.predict_proba(bow)[0]
        classes    = clf.classes_
        top3_idx   = np.argsort(probs)[::-1][:3]
        print("Top-3 classi:")
        for i in top3_idx:
            print(f"  {classes[i]:<25} {probs[i]*100:.1f}%")


if __name__ == "__main__":
    main()
