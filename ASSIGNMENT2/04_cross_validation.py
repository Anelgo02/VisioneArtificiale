"""
Fase 3 — Cross-validation: selezione del miglior modello.

Assi di confronto:
  1. Tipo di descrittore  : SIFT | ORB  (ORB opzionale: abilitare in config.py)
  2. Dimensione vocabolario: K ∈ {50, 100, 500}
  3. Classificatore       : SVM-RBF | Random Forest

Per ogni combinazione:
  - carica la rappresentazione BoW da models/bow_ucmerced_{desc_type}_K{k}.pkl
  - esegue 3-fold StratifiedKFold
  - calcola accuracy, precision, recall, F1-score (macro-averaged)

Output: results/cv_results.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import DESCRIPTOR_TYPES, K_VALUES, N_FOLDS, CV_RESULTS_FILE, RANDOM_SEED, bow_file
from utils import load_pickle, Timer


CLASSIFIERS = {
    "SVM-RBF":      SVC(kernel="rbf", random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
}


def evaluate_fold(clf, X_train, y_train, X_test, y_test) -> dict:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def cross_validate(clf_name: str, clf, X: np.ndarray, y: np.ndarray) -> dict:
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"    fold {fold}/{N_FOLDS} ...", end=" ", flush=True)
        fold_scores = evaluate_fold(
            clf,
            X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
        )
        scores.append(fold_scores)
        print(f"acc={fold_scores['accuracy']:.3f}  f1={fold_scores['f1']:.3f}")

    result = {"classifier": clf_name}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        vals = [s[metric] for s in scores]
        result[f"{metric}_mean"] = np.mean(vals)
        result[f"{metric}_std"]  = np.std(vals)

    return result


def main():
    if CV_RESULTS_FILE.exists():
        print(f"[skip] {CV_RESULTS_FILE} già presente — cancellalo per ricalcolare")
        sys.exit(0)

    all_results = []

    for desc_type in DESCRIPTOR_TYPES:
        for k in K_VALUES:
            bow_path = bow_file(desc_type, k)

            if not bow_path.exists():
                print(f"[skip] {bow_path.name} non trovato, salto")
                continue

            print(f"\n{'═'*55}")
            print(f"  {desc_type}  |  K = {k}")
            print(f"{'═'*55}")

            data = load_pickle(bow_path)
            X, y = data["X"], data["y"]
            print(f"  X: {X.shape}   classi: {len(set(y))}")

            for clf_name, clf in CLASSIFIERS.items():
                print(f"\n  ── {clf_name} ──")
                with Timer():
                    result = cross_validate(clf_name, clf, X, y)

                result["descriptor"] = desc_type
                result["K"]          = k
                all_results.append(result)

                print(f"  accuracy : {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}")
                print(f"  f1 macro : {result['f1_mean']:.3f} ± {result['f1_std']:.3f}")

    if not all_results:
        print("\n[ERR] Nessun risultato prodotto. Verifica che i file bow_ucmerced_*.pkl esistano.")
        sys.exit(1)

    cols = ["descriptor", "K", "classifier",
            "accuracy_mean", "accuracy_std",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std"]

    df = pd.DataFrame(all_results)[cols]
    df = df.sort_values(["descriptor", "K", "classifier"]).reset_index(drop=True)

    CV_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CV_RESULTS_FILE, index=False, float_format="%.4f")
    print(f"\n[saved] {CV_RESULTS_FILE}")

    print("\n── Riepilogo finale ──────────────────────────────────")
    print(df.to_string(index=False))

    best = df.loc[df["f1_mean"].idxmax()]
    print(f"\nModello migliore (F1 macro):")
    print(f"  descrittore  = {best['descriptor']}")
    print(f"  K            = {int(best['K'])}")
    print(f"  classificatore = {best['classifier']}")
    print(f"  accuracy     = {best['accuracy_mean']:.3f} ± {best['accuracy_std']:.3f}")
    print(f"  f1 macro     = {best['f1_mean']:.3f} ± {best['f1_std']:.3f}")
    print("\nFase 3 completata. Prossimo step: python 05_train_final.py")


if __name__ == "__main__":
    main()
