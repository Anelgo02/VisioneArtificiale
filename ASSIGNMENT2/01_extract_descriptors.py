"""
Fase 1a — Estrazione descrittori locali da AID.

Per ogni tipo di descrittore in DESCRIPTOR_TYPES:
  - itera su tutte le immagini AID
  - estrae descrittori (SIFT: float32 128-d  |  ORB: float32 32-d)
  - campiona al massimo MAX_DESCRIPTORS_PER_IMAGE per immagine
  - salva la matrice accumulata in models/descriptors_aid_{desc_type}.pkl

Output: models/descriptors_aid_SIFT.pkl  →  np.ndarray (N, 128) float32
        models/descriptors_aid_ORB.pkl   →  np.ndarray (N, 32)  float32  [se ORB abilitato]
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
    """Restituisce il rilevatore OpenCV per il tipo di descrittore richiesto."""
    if desc_type == "SIFT":
        return cv2.SIFT_create()
    if desc_type == "ORB":
        # nfeatures alto: campioneremo noi a max MAX_DESCRIPTORS_PER_IMAGE
        return cv2.ORB_create(nfeatures=500)
    raise ValueError(f"Tipo descrittore non supportato: {desc_type}")


def extract_descriptors(image_paths: list, desc_type: str,
                        max_per_image: int, seed: int) -> np.ndarray:
    extractor = make_extractor(desc_type)
    rng       = np.random.default_rng(seed)

    all_descriptors = []
    n_no_keypoints  = 0

    for path in tqdm(image_paths, desc=f"Estrazione {desc_type}", unit="img"):
        img = load_image_gray(path)
        if img is None:
            continue

        _, descs = extractor.detectAndCompute(img, None)

        if descs is None or len(descs) == 0:
            n_no_keypoints += 1
            continue

        if len(descs) > max_per_image:
            idx   = rng.choice(len(descs), size=max_per_image, replace=False)
            descs = descs[idx]

        # ORB restituisce uint8: convertiamo a float32 per K-Means euclideo
        all_descriptors.append(descs.astype(np.float32))

    if n_no_keypoints > 0:
        print(f"[warn] {n_no_keypoints} immagini senza keypoint {desc_type} (saltate)")

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

        if out_path.exists():
            print(f"[skip] {out_path.name} già presente")
            continue

        print(f"\n── {desc_type} ──────────────────────────────────────")
        print(f"Max descrittori per immagine: {MAX_DESCRIPTORS_PER_IMAGE}")

        with Timer():
            descriptors = extract_descriptors(
                image_paths, desc_type, MAX_DESCRIPTORS_PER_IMAGE, RANDOM_SEED
            )

        print(f"Descrittori estratti : {len(descriptors):,}")
        print(f"Shape matrice        : {descriptors.shape}")
        print(f"Memoria stimata      : {descriptors.nbytes / 1e6:.1f} MB")

        save_pickle(descriptors, out_path)

    print("\nFase 1a completata. Prossimo step: python 02_build_vocabulary.py")


if __name__ == "__main__":
    main()
