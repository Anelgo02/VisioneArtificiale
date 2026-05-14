"""
Fase 4 — Training finale sul modello vincente.

Legge cv_results.csv, identifica la combinazione migliore (descriptor, K, classificatore),
poi:
  1. Riaddestra il modello sul dataset UC Merced completo  → classifier_final.pkl
  2. Salva una copia del vocabolario vincente              → vocabulary_final.pkl
  3. Genera la matrice di confusione su un held-out 30%   → results/confusion_matrix.png

La matrice di confusione usa un singolo split 70/30 stratificato (solo per visualizzazione).
La metrica di riferimento rimane quella della cross-validation in cv_results.csv.
"""

import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

from config import (
    CV_RESULTS_FILE,
    FINAL_MODEL_FILE,
    FINAL_VOCAB_FILE,
    RESULTS_DIR,
    RANDOM_SEED,
    bow_file,
    vocabulary_file,
)
from utils import load_pickle, save_pickle, plot_confusion_matrix, Timer


CLASSIFIER_FACTORY = {
    "SVM-RBF":      lambda: SVC(kernel="rbf", random_state=RANDOM_SEED),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
}


def load_best_config() -> tuple[str, int, str]:
    if not CV_RESULTS_FILE.exists():
        print(f"[ERR] {CV_RESULTS_FILE} non trovato.")
        print("      Esegui prima: python 04_cross_validation.py")
        sys.exit(1)

    df   = pd.read_csv(CV_RESULTS_FILE)
    best = df.loc[df["f1_mean"].idxmax()]

    desc_type  = best["descriptor"]
    k          = int(best["K"])
    clf_name   = best["classifier"]

    print("── Configurazione vincente (da cv_results.csv) ──")
    print(f"  Descrittore  : {desc_type}")
    print(f"  K            : {k}")
    print(f"  Classificatore: {clf_name}")
    print(f"  F1 macro CV  : {best['f1_mean']:.3f} ± {best['f1_std']:.3f}")
    print(f"  Accuracy CV  : {best['accuracy_mean']:.3f} ± {best['accuracy_std']:.3f}")

    return desc_type, k, clf_name


def main():
    if FINAL_MODEL_FILE.exists():
        print(f"[skip] {FINAL_MODEL_FILE} già presente — cancellalo per riaddestrare")
        sys.exit(0)

    desc_type, k, clf_name = load_best_config()

    bow_path = bow_file(desc_type, k)
    if not bow_path.exists():
        print(f"\n[ERR] {bow_path} non trovato.")
        sys.exit(1)

    data        = load_pickle(bow_path)
    X           = data["X"]
    y           = data["y"]
    class_names = data["class_names"]

    print(f"\nX: {X.shape}   classi: {len(class_names)}")

    # ── 1. Matrice di confusione su split 70/30 ──────────────────────────────
    print("\n── Generazione matrice di confusione (split 70/30) ──")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X, y))

    clf_cm = CLASSIFIER_FACTORY[clf_name]()
    clf_cm.fit(X[train_idx], y[train_idx])
    y_pred = clf_cm.predict(X[test_idx])

    cm       = confusion_matrix(y[test_idx], y_pred, labels=class_names)
    cm_title = f"Confusion Matrix — {desc_type} K={k} {clf_name}"
    cm_path  = RESULTS_DIR / "confusion_matrix.png"

    plot_confusion_matrix(cm, class_names, title=cm_title, save_path=cm_path)

    acc_split = (y_pred == y[test_idx]).mean()
    print(f"Accuracy sullo split 30% : {acc_split:.3f}")

    # ── 2. Training finale su tutto il dataset ────────────────────────────────
    print("\n── Training finale su UC Merced completo ──")
    clf_final = CLASSIFIER_FACTORY[clf_name]()
    with Timer():
        clf_final.fit(X, y)

    save_pickle(clf_final, FINAL_MODEL_FILE)

    # ── 3. Copia vocabolario vincente come "vocabulary_final" ─────────────────
    voc_src = vocabulary_file(desc_type, k)
    shutil.copy(voc_src, FINAL_VOCAB_FILE)
    print(f"[saved] {FINAL_VOCAB_FILE}  (copia di {voc_src.name})")

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    print("\n── Riepilogo artefatti salvati ──")
    print(f"  {FINAL_MODEL_FILE}")
    print(f"  {FINAL_VOCAB_FILE}")
    print(f"  {cm_path}")
    print("\nFase 4 completata. Prossimo step: python inference.py <path_immagine>")


if __name__ == "__main__":
    main()
