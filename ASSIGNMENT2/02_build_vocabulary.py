"""
Fase 1b — Costruzione vocabolario visuale.

Per ogni combinazione (desc_type, K):
  - carica i descrittori AID da models/descriptors_aid_{desc_type}.pkl
  - esegue MiniBatchKMeans con K centroidi
  - salva il modello KMeans in models/vocabulary_{desc_type}_K{k}.pkl

Output: models/vocabulary_SIFT_K50.pkl, vocabulary_SIFT_K100.pkl, ...
        models/vocabulary_ORB_K50.pkl,  ...  [se ORB abilitato]

Cos'è il vocabolario visuale (BoW)?
    K-Means raggruppa ~1M descrittori SIFT in K cluster. Ogni centroide
    è una "visual word": un pattern visivo prototipico ricorrente nel dataset
    (es. un angolo, una texture, un bordo con una certa orientazione).
    Il vocabolario è il dizionario che, nella fase successiva, permette di
    codificare ogni immagine come istogramma di frequenza delle visual words.

Perché costruire il vocabolario solo su AID?
    Il vocabolario deve essere costruito su dati indipendenti dal test set
    (UC Merced) per evitare data leakage: se i centroidi fossero ottimizzati
    sui descrittori di test, l'istogramma BoW "vedrebbe" informazioni del test.
"""

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
    """
    Esegue MiniBatchKMeans sulla matrice dei descrittori e restituisce
    il modello addestrato (con k centroidi = k visual words).

    Parametri
    ---------
    descriptors : np.ndarray (N, D) float32
        Matrice di tutti i descrittori estratti da AID (fase 1a).
        N ≈ 1M per SIFT con max_per_image=100.
    k           : numero di cluster (dimensione del vocabolario)

    Perché MiniBatchKMeans invece di KMeans standard?
        KMeans standard aggiorna i centroidi su TUTTI i punti a ogni iterazione:
        con N=1M e D=128 ogni passo richiede ~512 MB di operazioni floating point
        e scala male col numero di iterazioni. MiniBatchKMeans aggiorna i centroidi
        su mini-batch casuali di KMEANS_BATCH_SIZE vettori: molto più veloce
        (10-100x) con perdita trascurabile di qualità dei centroidi.

    Scelta dei valori di K (K_VALUES = [50, 100, 500]):
        K piccolo  → vocabolario grossolano, istogrammi rumorosi, classificatore
                     veloce ma poco discriminativo.
        K grande   → vocabolario fine, maggiore potere discriminativo, ma più
                     costoso in quantizzazione e rischio di over-fitting sul
                     vocabolario stesso.
        I tre valori coprono un range significativo per confrontare l'impatto
        di K sulle metriche di classificazione (asse sperimentale richiesto).
    """
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        # Numero di vettori usati per aggiornare i centroidi a ogni passo.
        # Un batch più grande → convergenza più stabile ma più lenta.
        batch_size=KMEANS_BATCH_SIZE,
        # Numero massimo di passate sui mini-batch prima di fermarsi.
        # In pratica MiniBatchKMeans converge prima grazie all'early stopping interno.
        max_iter=KMEANS_MAX_ITER,
        # Seme fisso: garantisce che rieseguendo lo script si ottenga
        # esattamente lo stesso vocabolario (centroidi identici).
        random_state=RANDOM_SEED,
        verbose=0,
    )
    kmeans.fit(descriptors)

    # Salviamo l'intero oggetto KMeans (non solo kmeans.cluster_centers_) perché
    # nella fase BoW (03_compute_bow.py) useremo kmeans.predict() per assegnare
    # ogni nuovo descrittore alla visual word più vicina (nearest centroid).
    # Salvare solo i centroidi obbligherebbe a reimplementare la ricerca del NN.
    return kmeans


def main():
    for desc_type in DESCRIPTOR_TYPES:
        desc_path = descriptors_file(desc_type)

        # Dipende dalla fase 1a: se i descrittori non esistono
        # non ha senso procedere. Si stampa un messaggio orientativo invece
        # di sollevare un'eccezione, per gestire il caso in cui solo alcuni
        # tipi di descrittore siano stati estratti (es. SIFT sì, ORB no).
        if not desc_path.exists():
            print(f"[skip] {desc_path.name} non trovato — eseguire prima 01_extract_descriptors.py")
            continue

        print(f"\n{'═'*50}")
        print(f"  Descrittore: {desc_type}")
        print(f"{'═'*50}")
        print(f"Caricamento da {desc_path} ...")
        descriptors = load_pickle(desc_path)
        # La shape conferma quanti descrittori sono stati estratti e la loro
        # dimensionalità (128 per SIFT, 32 per ORB). Utile per debug.
        print(f"Shape: {descriptors.shape}  ({descriptors.nbytes / 1e6:.1f} MB)\n")

        for k in K_VALUES:
            out_path = vocabulary_file(desc_type, k)

            # Idempotenza: skip se il vocabolario per questa (desc_type, K) esiste già.
            # Per rigenerare un vocabolario specifico basta cancellare il .pkl corrispondente.
            if out_path.exists():
                print(f"[skip] K={k} — {out_path.name} già presente")
                continue

            print(f"── {desc_type}  K={k} ──────────────────────────────")
            with Timer():
                kmeans = build_vocabulary(descriptors, k)

            # Inertia = somma delle distanze quadratiche di ogni descrittore
            # dal centroide del proprio cluster (errore di quantizzazione totale).
            # Decresce al crescere di K: serve a confrontare vocabolari con K diverso,
            # non come metrica assoluta. Non è direttamente correlata all'accuracy
            # di classificazione: un K troppo alto può fare overfitting sul vocabolario.
            print(f"Inertia (errore di quantizzazione): {kmeans.inertia_:,.0f}")
            # n_iter_ = numero di iterazioni effettivamente eseguite prima della
            # convergenza (≤ KMEANS_MAX_ITER). Se uguale a MAX_ITER il modello
            # potrebbe non aver convergito: aumentare KMEANS_MAX_ITER in config.py.
            print(f"Iterazioni eseguite: {kmeans.n_iter_}")

            save_pickle(kmeans, out_path)
            print()

    print("Fase 1b completata. Prossimo step: python 03_compute_bow.py")


if __name__ == "__main__":
    main()
