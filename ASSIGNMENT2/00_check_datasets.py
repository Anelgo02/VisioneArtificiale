"""
Fase 0 — Verifica integrità dei dataset.
Eseguire prima di qualsiasi altra fase.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

from config import AID_ROOT, UCM_ROOT
from utils import SUPPORTED_EXTS, load_image_gray

# ── Colori ANSI ───────────────────────────────────────────────────────────────
OK   = "\033[92m[OK]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
ERR  = "\033[91m[ERR]\033[0m"

def _check_dataset(root: Path, name: str, expected_classes: int | None, expected_per_class: int | None) -> bool:
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"  Percorso: {root}")
    print(f"{'─'*55}")

    if not root.exists():
        print(f"  {ERR} Cartella non trovata.")
        return False

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"  {ERR} Nessuna sottocartella di classe trovata.")
        return False

    counts = {}
    broken = []
    for cls_dir in class_dirs:
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        counts[cls_dir.name] = len(imgs)

    n_classes = len(counts)
    total     = sum(counts.values())
    min_c     = min(counts.values())
    max_c     = max(counts.values())

    print(f"  Classi trovate : {n_classes}" +
          (f"  {WARN} attese {expected_classes}" if expected_classes and n_classes != expected_classes else f"  {OK}"))
    print(f"  Immagini totali: {total}")
    print(f"  Min/Max per classe: {min_c} / {max_c}" +
          (f"  {WARN} attese {expected_per_class}/classe" if expected_per_class and min_c != expected_per_class else ""))

    # Dettaglio per classe (mostra solo anomalie se tutto è uniforme)
    uniform = (min_c == max_c)
    if uniform:
        print(f"  {OK} Distribuzione uniforme ({min_c} immagini per classe)")
    else:
        print(f"  {WARN} Distribuzione non uniforme — dettaglio classi anomale:")
        mode_count = Counter(counts.values()).most_common(1)[0][0]
        for cls, n in sorted(counts.items()):
            if n != mode_count:
                print(f"       {cls}: {n} immagini  ← anomalia")

    # Test lettura su un campione (prima immagine di ogni classe)
    print(f"  Test lettura immagini (1 per classe)...")
    for cls_dir in class_dirs:
        imgs = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
        if not imgs:
            continue
        img = load_image_gray(imgs[0])
        if img is None:
            broken.append(imgs[0])

    if broken:
        print(f"  {ERR} {len(broken)} immagini non leggibili da OpenCV:")
        for p in broken[:5]:
            print(f"       {p}")
    else:
        print(f"  {OK} Tutte le immagini campionate sono leggibili da OpenCV")

    # Test SIFT su un campione
    sift = cv2.SIFT_create()
    first_img_path = sorted([p for p in class_dirs[0].iterdir() if p.suffix.lower() in SUPPORTED_EXTS])[0]
    img_test = load_image_gray(first_img_path)
    if img_test is not None:
        kps, descs = sift.detectAndCompute(img_test, None)
        if descs is not None and len(descs) > 0:
            print(f"  {OK} SIFT funzionante: {len(kps)} keypoint estratti da '{first_img_path.name}'")
        else:
            print(f"  {WARN} SIFT ha trovato 0 keypoint su '{first_img_path.name}' (immagine piatta?)")

    return len(broken) == 0 and n_classes > 0


def main():
    print("=" * 55)
    print("  VERIFICA DATASET — Assignment 2 BoW")
    print("=" * 55)

    ok_aid = _check_dataset(AID_ROOT,  "AID  (per vocabolario)",       expected_classes=30,  expected_per_class=None)
    ok_ucm = _check_dataset(UCM_ROOT,  "UC Merced  (per classificazione)", expected_classes=21, expected_per_class=100)

    print(f"\n{'='*55}")
    if ok_aid and ok_ucm:
        print(f"  {OK} Entrambi i dataset sono pronti. Puoi eseguire 01_extract_descriptors.py")
    else:
        print(f"  {ERR} Correggi gli errori segnalati prima di procedere.")
        sys.exit(1)
    print("=" * 55)


if __name__ == "__main__":
    main()
