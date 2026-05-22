"""
Fase 1b — Costruzione vocabolario visuale.

Per ogni combinazione (desc_type, K):
  - carica i descrittori AID da models/descriptors_aid_{desc_type}.pkl
  - esegue MiniBatchKMeans con K centroidi
  - salva il modello KMeans in models/vocabulary_{desc_type}_K{k}.pkl

Output: models/vocabulary_SIFT_K50.pkl, vocabulary_SIFT_K100.pkl, ...
        models/vocabulary_ORB_K50.pkl,  ...  [se ORB abilitato]
"""

import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config import (
    DESCRIPTOR_TYPES,
    K_VALUES,
    KMEANS_BATCH_SIZE,
    KMEANS_MAX_ITER,
    RANDOM_SEED,
    descriptors_file,
    vocabulary_file,
)
from utils import load_pickle, save_pickle, Timer


def build_vocabulary(descriptors: np.ndarray, k: int) -> MiniBatchKMeans:
    # MiniBatchKMeans trova K centroidi nello spazio a 128 dimensioni dei descrittori SIFT.
    # Ogni centroide diventa una "visual word": un pattern visivo prototipico.
    # Usiamo MiniBatch invece di KMeans standard perché con ~1M vettori da 128-d
    # il KMeans classico richiederebbe troppa RAM e tempo (aggiorna i centroidi
    # su mini-batch casuali anziché sull'intero dataset ad ogni iterazione).
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=KMEANS_BATCH_SIZE,
        max_iter=KMEANS_MAX_ITER,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    kmeans.fit(descriptors)
    # Salviamo l'intero oggetto KMeans (non solo kmeans.cluster_centers_) perché
    # in fase BoW useremo kmeans.predict() per assegnare ogni descrittore alla
    # visual word più vicina.
    return kmeans


def main():
    for desc_type in DESCRIPTOR_TYPES:
        desc_path = descriptors_file(desc_type)

        if not desc_path.exists():
            print(f"[skip] {desc_path.name} non trovato — eseguire prima 01_extract_descriptors.py")
            continue

        print(f"\n{'═'*50}")
        print(f"  Descrittore: {desc_type}")
        print(f"{'═'*50}")
        print(f"Caricamento da {desc_path} ...")
        descriptors = load_pickle(desc_path)
        print(f"Shape: {descriptors.shape}  ({descriptors.nbytes / 1e6:.1f} MB)\n")

        for k in K_VALUES:
            out_path = vocabulary_file(desc_type, k)

            if out_path.exists():
                print(f"[skip] K={k} — {out_path.name} già presente")
                continue

            print(f"── {desc_type}  K={k} ──────────────────────────────")
            with Timer():
                kmeans = build_vocabulary(descriptors, k)

            print(f"Inertia (errore di quantizzazione): {kmeans.inertia_:,.0f}")
            print(f"Iterazioni eseguite: {kmeans.n_iter_}")

            # salva l'intero oggetto KMeans 
            save_pickle(kmeans, out_path)
            print()

    print("Fase 1b completata. Prossimo step: python 03_compute_bow.py")


if __name__ == "__main__":
    main()
