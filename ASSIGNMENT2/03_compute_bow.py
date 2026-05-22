"""
Fase 2 — Rappresentazione Bag-of-Visual-Words per UC Merced.

Per ogni combinazione (desc_type, K):
  - carica il vocabolario da models/vocabulary_{desc_type}_K{k}.pkl
  - per ogni immagine UC Merced:
      1. estrae descrittori (SIFT o ORB)
      2. assegna ogni descrittore alla visual word più vicina (kmeans.predict)
      3. conta le occorrenze → istogramma di K bin
  - normalizza L2 tutta la matrice
  - salva X (2100 x K), y (2100,), class_names

Output: models/bow_ucmerced_{desc_type}_K{k}.pkl

Struttura dell'output salvato:
    {
        "X"           : np.ndarray (2100, K) float32  — feature matrix
        "y"           : np.ndarray (2100,)   str/int  — etichette di classe
        "class_names" : list[str]                     — nomi delle 21 classi
    }
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
    """
    Identica a quella in 01_extract_descriptors.py: restituisce il rilevatore
    OpenCV corretto per il tipo di descrittore. Duplicata qui per mantenere
    ogni script autonomo ed eseguibile indipendentemente.
    """
    if desc_type == "SIFT":
        return cv2.SIFT_create()
    if desc_type == "ORB":
        return cv2.ORB_create(nfeatures=500)
    raise ValueError(f"Tipo descrittore non supportato: {desc_type}")


def image_to_bow(img: np.ndarray, extractor, kmeans, k: int) -> np.ndarray:
    """
    Converte una singola immagine in un istogramma BoW di K bin.

    Pipeline interna:
        immagine grayscale
            → descrittori locali (N_kp x D)
            → hard assignment al centroide più vicino (N_kp indici in [0,K-1])
            → conteggio occorrenze → istogramma (K,) float32

    Il risultato NON è ancora normalizzato: la normalizzazione L2 viene
    applicata sull'intera matrice in compute_bow_matrix per efficienza.

    Parametri
    ---------
    img       : immagine in scala di grigi (np.ndarray uint8)
    extractor : oggetto SIFT o ORB già inizializzato
    kmeans    : modello MiniBatchKMeans caricato dalla fase 1b
    k         : numero di visual words (= n_clusters del kmeans)
    """
    _, descs = extractor.detectAndCompute(img, None)

    # Immagine senza keypoint (es. zona piatta senza gradienti): restituiamo
    # un istogramma di zeri. Dopo la normalizzazione L2 resterà zero: il
    # classificatore lo tratterà come un campione "neutro".
    # Questo caso viene contato in compute_bow_matrix per avvisare l'utente.
    if descs is None or len(descs) == 0:
        return np.zeros(k, dtype=np.float32)

    # Hard assignment: ogni descrittore viene assegnato alla visual word più vicina
    # (il centroide KMeans più prossimo in senso euclideo). predict() restituisce
    # un array di indici interi in [0, K-1], uno per ogni riga di descs.
    # Si chiama "hard" perché ogni descrittore va in un unico cluster (a differenza
    # del "soft assignment" che distribuisce il peso su più centroidi vicini).
    words = kmeans.predict(descs.astype(np.float32))

    # np.bincount conta quante volte compare ogni intero in [0, K-1]:
    #   words = [3, 0, 3, 7, 3]  con K=10  →  hist[0]=1, hist[3]=3, hist[7]=1, rest=0
    # minlength=k garantisce che l'array abbia sempre esattamente K elementi
    # anche se alcune visual words non compaiono nell'immagine (bin vuoti = 0).
    hist = np.bincount(words, minlength=k).astype(np.float32)
    return hist


def compute_bow_matrix(paths, labels, desc_type, kmeans, k):
    """
    Costruisce la feature matrix X e il vettore etichette y per tutte le immagini.

    Parametri
    ---------
    paths     : lista di Path alle immagini UC Merced (ordine fisso)
    labels    : lista di etichette corrispondenti (stesso ordine di paths)
    desc_type : "SIFT" o "ORB"
    kmeans    : modello MiniBatchKMeans caricato
    k         : dimensione del vocabolario

    Ritorna
    -------
    X : np.ndarray (n, k) float32 — istogrammi BoW L2-normalizzati
    y : np.ndarray (n,)           — etichette di classe
    """
    extractor      = make_extractor(desc_type)
    n              = len(paths)

    # Preallochiamo la matrice intera a zero prima del loop:
    # - evita reallocazioni incrementali (append + vstack) che costano O(n²)
    # - le righe corrispondenti a immagini senza keypoint rimangono zero
    #   senza bisogno di gestirle separatamente dopo il loop
    X              = np.zeros((n, k), dtype=np.float32)
    n_no_keypoints = 0

    for i, path in enumerate(tqdm(paths, desc=f"BoW {desc_type} K={k}", unit="img")):
        img = load_image_gray(path)
        if img is None:
            # Immagine non leggibile: la riga i di X resta zero.
            # Segnalato da 00_check_datasets.py; qui si prosegue silenziosamente.
            continue
        hist = image_to_bow(img, extractor, kmeans, k)
        if hist.sum() == 0:
            # Immagine leggibile ma senza keypoint: istogramma tutto zero.
            # Contato separatamente per il warning finale.
            n_no_keypoints += 1
        X[i] = hist

    if n_no_keypoints > 0:
        print(f"[warn] {n_no_keypoints} immagini senza keypoint (istogramma zero)")

    # Normalizzazione L2 applicata sull'intera matrice in un'unica operazione
    X = normalize(X, norm="l2")
    y = np.array(labels)
    return X, y


def main():
    print(f"Dataset UC Merced: {UCM_ROOT}")

    # collect_labeled_paths ritorna due liste parallele:
    #   paths  = [Path("…/agricultural/img01.tif"), …]
    #   labels = ["agricultural", …]  (una etichetta per ogni path)
    paths, labels = collect_labeled_paths(UCM_ROOT)
    class_names   = sorted(set(labels))  # 21 nomi di classe in ordine alfabetico

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

            # Idempotenza: skip se l'output esiste già.
            if out_path.exists():
                print(f"[skip] {out_path.name} già presente")
                continue

            # Dipendenza dalla fase 1b: senza vocabolario non si può quantizzare.
            # Si salta (invece di uscire) perché altri valori di K potrebbero
            # avere il vocabolario disponibile.
            if not voc_path.exists():
                print(f"[skip] {voc_path.name} non trovato, salto")
                continue

            kmeans = load_pickle(voc_path)

            with Timer():
                X, y = compute_bow_matrix(paths, labels, desc_type, kmeans, k)

            print(f"Shape X  : {X.shape}")
            # Sanity check: dopo normalize(norm="l2") ogni riga dovrebbe avere
            # norma euclidea = 1.0 (tolleranza float32 ≈ 1e-6).
            # Se il valore è 0.0 la prima immagine non aveva keypoint.
            print(f"Norma L2 (prima riga): {np.linalg.norm(X[0]):.4f}  (atteso ~1.0)")
            save_pickle({"X": X, "y": y, "class_names": class_names}, out_path)
            print()

    print("Fase 2 completata. Prossimo step: python 04_cross_validation.py")


if __name__ == "__main__":
    main()
