"""
Fase 0 — Verifica integrità dei dataset.
Eseguire prima di qualsiasi altra fase.

Controlla che AID e UC Merced siano presenti su disco, abbiano il numero
atteso di classi e di immagini per classe, che tutte le immagini siano
leggibili da OpenCV e che SIFT riesca ad estrarre keypoint su almeno
un campione. Termina con exit(1) se un dataset è corrotto o mancante,
così i passi successivi non vengono eseguiti su dati inconsistenti.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

from config import AID_ROOT, UCM_ROOT
from utils import SUPPORTED_EXTS, load_image_gray

# ── Colori ANSI ───────────────────────────────────────────────────────────────
# Prefissi colorati per distinguere rapidamente esito OK / avviso / errore
# nel terminale senza dover leggere l'intero messaggio.
OK   = "\033[92m[OK]\033[0m"    # verde
WARN = "\033[93m[WARN]\033[0m"  # giallo
ERR  = "\033[91m[ERR]\033[0m"   # rosso


def _check_dataset(root: Path, name: str, expected_classes: int | None, expected_per_class: int | None) -> bool:
    """
    Esegue i controlli di integrità su un singolo dataset organizzato come:
        root/
            <classe_1>/img1.jpg  img2.jpg  ...
            <classe_2>/img1.jpg  ...

    Parametri
    ---------
    root               : percorso alla cartella radice del dataset
    name               : etichetta descrittiva stampata nell'intestazione
    expected_classes   : numero di classi atteso (None = non controllato)
    expected_per_class : numero di immagini atteso per ogni classe (None = non controllato)

    Ritorna True se il dataset è usabile (almeno 1 classe, 0 immagini corrotte),
    False in caso di errori bloccanti.
    """
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"  Percorso: {root}")
    print(f"{'─'*55}")

    # ── 1. Verifica esistenza della cartella radice ────────────────────────────
    if not root.exists():
        print(f"  {ERR} Cartella non trovata.")
        return False

    # ── 2. Raccolta delle sottocartelle di classe ──────────────────────────────
    # Ogni sottocartella diretta di `root` è trattata come una classe.
    # I file nella radice stessa vengono ignorati.
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"  {ERR} Nessuna sottocartella di classe trovata.")
        return False

    # ── 3. Conteggio immagini per classe ──────────────────────────────────────
    # Considera solo le estensioni in SUPPORTED_EXTS (es. .jpg, .tif, .png)
    # per escludere file ausiliari come .txt, .DS_Store ecc.
    counts = {}
    broken = []  # raccoglie i path delle immagini non leggibili (riempito dopo)
    for cls_dir in class_dirs:
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        counts[cls_dir.name] = len(imgs)

    n_classes = len(counts)
    total     = sum(counts.values())
    min_c     = min(counts.values())
    max_c     = max(counts.values())

    # ── 4. Stampa statistiche aggregate ───────────────────────────────────────
    # Se il conteggio classi differisce dall'atteso, viene emesso un WARN
    # (non un ERR: il dataset potrebbe comunque essere usabile parzialmente).
    print(f"  Classi trovate : {n_classes}" +
          (f"  {WARN} attese {expected_classes}" if expected_classes and n_classes != expected_classes else f"  {OK}"))
    print(f"  Immagini totali: {total}")
    print(f"  Min/Max per classe: {min_c} / {max_c}" +
          (f"  {WARN} attese {expected_per_class}/classe" if expected_per_class and min_c != expected_per_class else ""))

    # ── 5. Verifica distribuzione delle classi ─────────────────────────────────
    # Se tutte le classi hanno lo stesso numero di immagini il dataset è
    # bilanciato: basta una riga. Altrimenti si calcolano le anomalie
    # rispetto alla moda (valore più frequente), così si evidenziano solo
    # le classi con conteggio fuori dalla norma.
    uniform = (min_c == max_c)
    if uniform:
        print(f"  {OK} Distribuzione uniforme ({min_c} immagini per classe)")
    else:
        print(f"  {WARN} Distribuzione non uniforme — dettaglio classi anomale:")
        mode_count = Counter(counts.values()).most_common(1)[0][0]
        for cls, n in sorted(counts.items()):
            if n != mode_count:
                print(f"       {cls}: {n} immagini  ← anomalia")

    # ── 6. Test lettura con OpenCV (campione: 1 immagine per classe) ───────────
    # load_image_gray restituisce None se OpenCV non riesce ad aprire il file
    # (es. file troncato, formato non supportato, path con caratteri Unicode
    # su Windows). Leggere tutte le immagini sarebbe lento: un campione per
    # classe è sufficiente a rilevare problemi di configurazione sistematici.
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
        for p in broken[:5]:  # mostra al massimo 5 path per non intasare l'output
            print(f"       {p}")
    else:
        print(f"  {OK} Tutte le immagini campionate sono leggibili da OpenCV")

    # ── 7. Test SIFT su un campione ────────────────────────────────────────────
    # Verifica che SIFT sia disponibile nella build di OpenCV installata
    # (assente in alcuni package cv2 senza contrib) e che l'immagine campione
    # non sia troppo uniforme da non produrre keypoint. Un'immagine piatta
    # (es. tutto bianco/nero) non darebbe descrittori utili per il vocabolario.
    sift = cv2.SIFT_create()
    first_img_path = sorted([p for p in class_dirs[0].iterdir() if p.suffix.lower() in SUPPORTED_EXTS])[0]
    img_test = load_image_gray(first_img_path)
    if img_test is not None:
        kps, descs = sift.detectAndCompute(img_test, None)
        if descs is not None and len(descs) > 0:
            print(f"  {OK} SIFT funzionante: {len(kps)} keypoint estratti da '{first_img_path.name}'")
        else:
            print(f"  {WARN} SIFT ha trovato 0 keypoint su '{first_img_path.name}' (immagine piatta?)")

    # Il dataset è considerato "ok" se non ci sono immagini corrotte e
    # almeno una classe è presente (n_classes > 0 è già garantito qui sopra).
    return len(broken) == 0 and n_classes > 0


def main():
    print("=" * 55)
    print("  VERIFICA DATASET — Assignment 2 BoW")
    print("=" * 55)

    # AID: 30 classi, numero immagini per classe variabile (non uniforme)
    # → expected_per_class=None per non emettere falsi WARN
    ok_aid = _check_dataset(AID_ROOT,  "AID  (per vocabolario)",       expected_classes=30,  expected_per_class=None)

    # UC Merced: 21 classi, esattamente 100 immagini per classe (dataset bilanciato)
    ok_ucm = _check_dataset(UCM_ROOT,  "UC Merced  (per classificazione)", expected_classes=21, expected_per_class=100)

    print(f"\n{'='*55}")
    if ok_aid and ok_ucm:
        print(f"  {OK} Entrambi i dataset sono pronti. Puoi eseguire 01_extract_descriptors.py")
    else:
        # Uscita con codice non-zero per bloccare eventuali script shell o
        # Makefile che concatenano i passi della pipeline con &&.
        print(f"  {ERR} Correggi gli errori segnalati prima di procedere.")
        sys.exit(1)
    print("=" * 55)


if __name__ == "__main__":
    main()
