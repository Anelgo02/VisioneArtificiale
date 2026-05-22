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

Perché StratifiedKFold e non KFold?
    KFold semplice potrebbe creare fold sbilanciati (un fold con più campioni
    di "agricultural" e meno di "storagetanks"). StratifiedKFold garantisce
    che ogni fold abbia la stessa proporzione di classi del dataset completo,
    rendendo le metriche per fold comparabili tra loro.
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


# Dizionario dei classificatori da confrontare.
# SVM-RBF: kernel gaussiano, adatto a feature vettoriali dense come gli istogrammi
#   BoW L2-normalizzati. Efficace in spazi ad alta dimensionalità (K=500 → 500-d).
# RandomForest: ensemble di alberi decisionali, lavora bene senza normalizzazione
#   ma qui riceverebbe già vettori normalizzati. Serve come baseline alternativa
#   per valutare se un metodo non-kernel compete con SVM su questi dati.
# n_jobs=-1 in RF: usa tutti i core disponibili per parallelizzare la costruzione
#   degli alberi (ogni albero è indipendente dagli altri).
CLASSIFIERS = {
    "SVM-RBF":      SVC(kernel="rbf", random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
}


def evaluate_fold(clf, X_train, y_train, X_test, y_test) -> dict:
    """
    Addestra il classificatore su un fold e restituisce le metriche sul test.

    Parametri
    ---------
    clf              : classificatore scikit-learn (già istanziato)
    X_train, y_train : dati di training del fold corrente
    X_test,  y_test  : dati di test del fold corrente

    Ritorna
    -------
    dict con chiavi: accuracy, precision, recall, f1
        Tutte le metriche multi-classe usano average="macro":
        si calcola la metrica per ogni classe e si fa la media non pesata.
        Con 21 classi bilanciate (100 campioni ciascuna) macro e weighted
        coincidono, ma macro è più conservativo: penalizza di più gli errori
        sulle classi piccole (se presenti).
        zero_division=0: se una classe non appare nelle predizioni, la sua
        precision/recall vale 0 anziché generare un warning.
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def cross_validate(clf_name: str, clf, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Esegue N_FOLDS-fold stratificata e aggrega le metriche (media ± std).

    Parametri
    ---------
    clf_name : nome del classificatore (usato solo per il campo "classifier" nell'output)
    clf      : istanza del classificatore (viene ri-addestrata da zero a ogni fold)
    X        : feature matrix (n_samples, K) float32
    y        : etichette (n_samples,)

    Ritorna
    -------
    dict con: classifier, accuracy_mean/std, precision_mean/std,
              recall_mean/std, f1_mean/std
    """
    # StratifiedKFold garantisce che ogni fold abbia la stessa proporzione di classi
    # del dataset originale
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

    # Aggregazione: per ogni metrica calcoliamo media e deviazione standard sui fold.
    # La std misura la variabilità della metrica al variare del fold: una std alta
    # indica che le prestazioni dipendono molto da quali campioni finiscono nel test,
    # ovvero che il modello è instabile o il dataset ha alta varianza inter-fold.
    result = {"classifier": clf_name}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        vals = [s[metric] for s in scores]
        result[f"{metric}_mean"] = np.mean(vals)
        result[f"{metric}_std"]  = np.std(vals)

    return result


def main():
    # Idempotenza a livello di file: se il CSV esiste già, lo script esce senza
    # ricalcolare. A differenza degli altri script, qui non si può fare skip
    # per singola combinazione (il CSV è un unico file), quindi si blocca tutto.
    # Per ricalcolare: cancellare results/cv_results.csv.
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

                # Aggiunta delle colonne di contesto al risultato del fold:
                # necessarie per distinguere le righe nel CSV finale.
                result["descriptor"] = desc_type
                result["K"]          = k
                all_results.append(result)

                print(f"  accuracy : {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}")
                print(f"  f1 macro : {result['f1_mean']:.3f} ± {result['f1_std']:.3f}")

    if not all_results:
        print("\n[ERR] Nessun risultato prodotto. Verifica che i file bow_ucmerced_*.pkl esistano.")
        sys.exit(1)

    # Selezione e ordinamento delle colonne per leggibilità del CSV:
    # prima le chiavi di identificazione (descriptor, K, classifier),
    # poi le metriche in ordine logico (accuracy → f1).
    cols = ["descriptor", "K", "classifier",
            "accuracy_mean", "accuracy_std",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std"]

    df = pd.DataFrame(all_results)[cols]
    # Ordinamento per rendere il CSV navigabile: raggruppa per descrittore,
    # poi per K crescente, poi per nome classificatore alfabetico.
    df = df.sort_values(["descriptor", "K", "classifier"]).reset_index(drop=True)

    CV_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # float_format="%.4f": 4 cifre decimali sono sufficienti per metriche in [0,1];
    # evita la notazione scientifica che renderebbe il CSV meno leggibile.
    df.to_csv(CV_RESULTS_FILE, index=False, float_format="%.4f")
    print(f"\n[saved] {CV_RESULTS_FILE}")

    print("\n── Riepilogo finale ──────────────────────────────────")
    print(df.to_string(index=False))

    # Il modello vincente è quello con F1 macro medio più alto.
    # F1 macro è la metrica principale perché bilancia precision e recall
    # su tutte le classi equipesate, indipendentemente dalla loro frequenza.
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
