"""
Fase 1a — Estrazione descrittori locali da AID.

Per ogni tipo di descrittore in DESCRIPTOR_TYPES:
  - itera su tutte le immagini AID
  - estrae descrittori (SIFT: float32 128-d  |  ORB: float32 32-d)
  - campiona al massimo MAX_DESCRIPTORS_PER_IMAGE per immagine
  - salva la matrice accumulata in models/descriptors_aid_{desc_type}.pkl

Output: models/descriptors_aid_SIFT.pkl  →  np.ndarray (N, 128) float32
        models/descriptors_aid_ORB.pkl   →  np.ndarray (N, 32)  float32  [se ORB abilitato]

Perché solo AID e non UC Merced?
    Il vocabolario BoW deve essere costruito su un insieme di immagini
    indipendente dal test set. AID è il dataset usato per il training del
    vocabolario; UC Merced è riservato alla classificazione (fase 3).
    Estrarre i descrittori su AID evita data leakage nel vocabolario.
"""

import sys
import numpy as np
import cv2
from tqdm import tqdm

from config import (
    AID_ROOT,
    DESCRIPTOR_TYPES,
    MAX_DESCRIPTORS_PER_IMAGE,
    RANDOM_SEED,
    descriptors_file,
)
from utils import collect_image_paths, load_image_gray, save_pickle, Timer


def make_extractor(desc_type: str):
    """
    Costruisce e restituisce il rilevatore/descrittore OpenCV corretto.

    SIFT  → cv2.SIFT_create()
        Descrittore float32 a 128 dimensioni, invariante a scala e rotazione.
        Richiede opencv-contrib-python oppure opencv-python >= 4.4 (brevetto scaduto).

    ORB   → cv2.ORB_create(nfeatures=500)
        Descrittore binario a 256 bit (32 byte), molto più veloce di SIFT
        ma meno discriminativo. nfeatures=500 è impostato alto perché il
        campionamento a MAX_DESCRIPTORS_PER_IMAGE viene fatto manualmente
        dopo, per coerenza con la logica SIFT.
        NOTA: K-Means usa distanza euclidea, non Hamming → i descrittori ORB
        vengono convertiti in float32, che è un'approssimazione (vedi fase di
        estrazione). Dichiarare questa scelta se si includono risultati ORB.
    """
    if desc_type == "SIFT":
        return cv2.SIFT_create()
    if desc_type == "ORB":
        # nfeatures alto: campioneremo noi a max MAX_DESCRIPTORS_PER_IMAGE
        return cv2.ORB_create(nfeatures=500)
    raise ValueError(f"Tipo descrittore non supportato: {desc_type}")


def extract_descriptors(image_paths: list, desc_type: str,
                        max_per_image: int, seed: int) -> np.ndarray:
    """
    Estrae e accumula i descrittori da una lista di immagini.

    Parametri
    ---------
    image_paths  : lista di Path alle immagini da processare
    desc_type    : "SIFT" o "ORB"
    max_per_image: numero massimo di descrittori tenuti per immagine
    seed         : seme per il campionamento casuale (riproducibilità)

    Ritorna
    -------
    np.ndarray di shape (N_totale, D) float32
        N_totale ≤ len(image_paths) × max_per_image
        D = 128 per SIFT, 32 per ORB
    """
    extractor = make_extractor(desc_type)

    # Generatore NumPy con seme fisso: garantisce che campionamenti diversi
    # su macchine diverse producano sempre la stessa selezione di descrittori,
    # rendendo il vocabolario riproducibile.
    rng = np.random.default_rng(seed)

    all_descriptors = []
    n_no_keypoints  = 0

    for path in tqdm(image_paths, desc=f"Estrazione {desc_type}", unit="img"):
        img = load_image_gray(path)
        if img is None:
            # load_image_gray restituisce None se OpenCV non riesce ad aprire
            # il file (path Unicode su Windows, file troncato, ecc.). Si salta
            # silenziosamente: 00_check_datasets.py segnala già queste immagini.
            continue

        # detectAndCompute combina detezione e descrizione in un'unica passata
        # sull'immagine, più efficiente che chiamare detect() e compute() separati.
        #   kps   → lista di cv2.KeyPoint (posizione, scala, orientamento)
        #   descs → np.ndarray (N_kp, D):  una riga per keypoint
        # Scartiamo kps (il '_'): per il BoW servono solo i vettori descrittori,
        # non le coordinate spaziali dei keypoint.
        _, descs = extractor.detectAndCompute(img, None)

        if descs is None or len(descs) == 0:
            # Accade su immagini molto uniformi (cieli bianchi, aree piatte).
            # SIFT e ORB non trovano strutture locali su cui calcolare il descrittore.
            n_no_keypoints += 1
            continue

        if len(descs) > max_per_image:
            # Campionamento casuale senza reinserimento (replace=False):
            # scegliamo max_per_image indici tra 0..N_kp-1 e prendiamo le
            # righe corrispondenti della matrice descrittori.
            #
            # Perché limitare a 100 desc/immagine?
            #   - AID ha ~10.000 immagini → 10.000 × 100 ≈ 1M vettori
            #   - 1M × 128 float32 ≈ 512 MB in RAM: gestibile da MiniBatchKMeans
            #   - Ridurre il numero preserva la diversità del vocabolario:
            #     tenere tutti i desc farebbe dominare le immagini con molti
            #     keypoint (es. zone urbane dense) rispetto ad altre categorie.
            idx   = rng.choice(len(descs), size=max_per_image, replace=False)
            descs = descs[idx]

        # ORB produce descrittori uint8 (ogni byte è una maschera di bit Hamming).
        # K-Means minimizza la distanza euclidea L2, definita solo su float.
        # La conversione uint8→float32 è un'approssimazione: tratta i byte come
        # numeri interi continui anziché come vettori binari. È accettabile come
        # baseline ma inferiore a metodi clustering binari (es. k-medoids + Hamming).
        all_descriptors.append(descs.astype(np.float32))

    if n_no_keypoints > 0:
        print(f"[warn] {n_no_keypoints} immagini senza keypoint {desc_type} (saltate)")

    # np.vstack impila verticalmente le matrici di tutte le immagini:
    #   [(N1,D), (N2,D), ..., (Nk,D)]  →  (N1+N2+...+Nk, D)
    # Il risultato è la matrice che entra in MiniBatchKMeans nella fase 2
    # per costruire il vocabolario visuale (i centroidi = "visual words").
    return np.vstack(all_descriptors)


def main():
    print(f"Dataset AID: {AID_ROOT}")
    image_paths = collect_image_paths(AID_ROOT)
    print(f"Immagini trovate: {len(image_paths)}")

    if len(image_paths) == 0:
        print("[ERR] Nessuna immagine trovata. Verifica il percorso in config.py")
        sys.exit(1)

    for desc_type in DESCRIPTOR_TYPES:
        out_path = descriptors_file(desc_type)

        # Idempotenza: se il file .pkl esiste già, il passo viene saltato.
        # Per forzare il ricalcolo basta cancellare il file corrispondente in models/.
        if out_path.exists():
            print(f"[skip] {out_path.name} già presente")
            continue

        print(f"\n── {desc_type} ──────────────────────────────────────")
        print(f"Max descrittori per immagine: {MAX_DESCRIPTORS_PER_IMAGE}")

        # Timer è un context manager che misura e stampa il tempo trascorso;
        # utile per stimare la durata del passo su macchine diverse.
        with Timer():
            descriptors = extract_descriptors(
                image_paths, desc_type, MAX_DESCRIPTORS_PER_IMAGE, RANDOM_SEED
            )

        # Informazioni diagnostiche sulla matrice risultante:
        #   - len(descriptors) = numero totale di vettori accumulati
        #   - descriptors.shape = (N_totale, D), es. (1_000_000, 128) per SIFT
        #   - nbytes / 1e6 = dimensione in MB (utile per stimare uso RAM in fase 2)
        print(f"Descrittori estratti : {len(descriptors):,}")
        print(f"Shape matrice        : {descriptors.shape}")
        print(f"Memoria stimata      : {descriptors.nbytes / 1e6:.1f} MB")

        save_pickle(descriptors, out_path)

    print("\nFase 1a completata. Prossimo step: python 02_build_vocabulary.py")


if __name__ == "__main__":
    main()
