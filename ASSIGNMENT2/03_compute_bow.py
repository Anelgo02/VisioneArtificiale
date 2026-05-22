"""
Fase 2 — Rappresentazione Bag-of-Visual-Words per UC Merced.

Per ogni combinazione (desc_type, K):
  - carica il vocabolario da models/vocabulary_{desc_type}_K{k}.pkl
  - per ogni immagine UC Merced:
      1. estrae descrittori (SIFT o ORB)
      2. assegna ogni descrittore alla visual word più vicina (kmeans.predict)
      3. conta le occorrenze → istogramma di K bin
  - normalizza L2 tutta la matrice
  - salva X (2100 × K), y (2100,), class_names

Output: models/bow_ucmerced_{desc_type}_K{k}.pkl
"""

import sys
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from tqdm import tqdm

from config import (
    UCM_ROOT,
    DESCRIPTOR_TYPES,
    K_VALUES,
    vocabulary_file,
    bow_file,
)
from utils import collect_labeled_paths, load_image_gray, load_pickle, save_pickle, Timer


def make_extractor(desc_type: str):
    if desc_type == "SIFT":
        return cv2.SIFT_create()
    if desc_type == "ORB":
        return cv2.ORB_create(nfeatures=500)
    raise ValueError(f"Tipo descrittore non supportato: {desc_type}")


def image_to_bow(img: np.ndarray, extractor, kmeans, k: int) -> np.ndarray:
    _, descs = extractor.detectAndCompute(img, None)

    # Immagine senza keypoint (es. zona piatta senza gradienti): restituiamo
    # un istogramma di zeri. Dopo la normalizzazione L2 resterà zero: il
    # classificatore lo tratterà come un campione "neutro".
    if descs is None or len(descs) == 0:
        return np.zeros(k, dtype=np.float32)

    # Hard assignment: ogni descrittore viene assegnato alla visual word più vicina
    # (il centroide KMeans più prossimo in senso euclideo). predict() restituisce
    # un array di indici interi in [0, K-1], uno per ogni riga di descs.
    words = kmeans.predict(descs.astype(np.float32))

    # Conta quante volte compare ogni visual word → istogramma di K bin.
    # minlength=k garantisce che l'array abbia sempre K elementi anche se
    # alcune visual words non compaiono nell'immagine.
    hist  = np.bincount(words, minlength=k).astype(np.float32)
    return hist


def compute_bow_matrix(paths, labels, desc_type, kmeans, k):
    extractor      = make_extractor(desc_type)
    n              = len(paths)
    X              = np.zeros((n, k), dtype=np.float32) # preallochiamo tutte le 2100 righe per evitare disallineamenti dovuti a immagini senza keypoint
    n_no_keypoints = 0

    for i, path in enumerate(tqdm(paths, desc=f"BoW {desc_type} K={k}", unit="img")):
        img = load_image_gray(path)
        if img is None:
            continue
        hist = image_to_bow(img, extractor, kmeans, k)
        if hist.sum() == 0:
            n_no_keypoints += 1
        X[i] = hist

    if n_no_keypoints > 0:
        print(f"[warn] {n_no_keypoints} immagini senza keypoint (istogramma zero)")

    # Normalizzazione L2: porta ogni istogramma a norma unitaria.
    # Senza normalizzazione, immagini con molti keypoint avrebbero istogrammi
    # con valori assoluti maggiori, falsando la distanza percepita dall'SVM.
    # Con L2, il classificatore confronta la *distribuzione relativa* delle visual words.
    X = normalize(X, norm="l2")
    y = np.array(labels)
    return X, y


def main():
    print(f"Dataset UC Merced: {UCM_ROOT}")
    paths, labels = collect_labeled_paths(UCM_ROOT)
    class_names   = sorted(set(labels))

    print(f"Immagini trovate : {len(paths)}")
    print(f"Classi           : {len(class_names)}")

    if len(paths) == 0:
        print("[ERR] Nessuna immagine trovata. Verifica il percorso in config.py")
        sys.exit(1)

    for desc_type in DESCRIPTOR_TYPES:
        print(f"\n{'═'*50}")
        print(f"  Descrittore: {desc_type}")
        print(f"{'═'*50}")

        for k in K_VALUES:
            out_path = bow_file(desc_type, k)
            voc_path = vocabulary_file(desc_type, k)

            if out_path.exists():
                print(f"[skip] {out_path.name} già presente")
                continue

            if not voc_path.exists():
                print(f"[skip] {voc_path.name} non trovato, salto")
                continue

            kmeans = load_pickle(voc_path)

            with Timer():
                X, y = compute_bow_matrix(paths, labels, desc_type, kmeans, k)

            print(f"Shape X  : {X.shape}")
            print(f"Norma L2 (prima riga): {np.linalg.norm(X[0]):.4f}  (atteso ~1.0)")
            save_pickle({"X": X, "y": y, "class_names": class_names}, out_path)
            print()

    print("Fase 2 completata. Prossimo step: python 04_cross_validation.py")


if __name__ == "__main__":
    main()
