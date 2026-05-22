"""
Fase 4 — Training finale sul modello vincente.

Legge cv_results.csv, identifica la combinazione migliore (descriptor, K, classificatore),
poi:
  1. Riaddestra il modello sul dataset UC Merced completo  → classifier_final.pkl
  2. Salva una copia del vocabolario vincente              → vocabulary_final.pkl
  3. Genera la matrice di confusione su un held-out 30%   → results/confusion_matrix.png

La matrice di confusione usa un singolo split 70/30 stratificato (solo per visualizzazione).
La metrica di riferimento rimane quella della cross-validation in cv_results.csv.

Perché riaddestrare su tutto il dataset dopo la CV?
    Durante la cross-validation ogni fold usa solo il 66% dei dati per il training.
    Il modello finale di deployment deve essere il più generalizzabile possibile,
    quindi si ri-addestra su tutti i 2100 campioni. Le metriche di riferimento
    (accuracy, F1) rimangono quelle della CV, non quelle dello split 70/30 che
    serve solo per visualizzare la matrice di confusione.

Perché la CM usa uno split 70/30 separato e non la CV?
    La CV produce N predizioni per fold disgiunti, ma ricomporle in una singola
    CM richiederebbe di concatenare i y_pred dei 3 fold — pratica valida ma che
    mescola modelli addestrati su dati diversi. Lo split 70/30 stratificato è
    più semplice e produce una CM pulita su un singolo modello coerente.
    Il modello addestrato sul 70% viene scartato dopo la CM: non è il finale.
"""

import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report

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


# Factory di classificatori: usa lambda invece di istanze dirette (come in 04)
# per garantire che ogni chiamata crei un oggetto fresco senza stato residuo
# da addestramenti precedenti. In questo script il classificatore viene
# istanziato due volte (una per la CM, una per il finale), quindi è
# importante partire da zero ogni volta.
CLASSIFIER_FACTORY = {
    "SVM-RBF":      lambda: SVC(kernel="rbf", random_state=RANDOM_SEED),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
}


def load_best_config() -> tuple[str, int, str]:
    """
    Legge cv_results.csv e restituisce la configurazione con F1 macro più alto.

    Ritorna
    -------
    (desc_type, k, clf_name) : tipo descrittore, dimensione vocabolario, nome classificatore

    La selezione si basa su f1_mean (media sui fold) e non su accuracy_mean
    perché F1 macro penalizza i classificatori che ignorano le classi difficili.
    Con 21 classi bilanciate le due metriche sono molto correlate, ma F1 è
    più informativo in generale.
    """
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
    # Idempotenza: se il modello finale esiste già, lo script esce.
    # Per riaddestrare: cancellare models/classifier_final.pkl.
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
    # La CM mostra quali classi vengono confuse tra loro su dati mai visti.
    # StratifiedShuffleSplit con n_splits=1 produce un singolo split 70/30
    # stratificato: ogni classe ha ~70 campioni in train e ~30 in test.
    # Stratificato → le proporzioni di classe sono identiche in train e test,
    # evitando che la CM sia distorta da classi sottorappresentate nel test.
    print("\n── Generazione matrice di confusione (split 70/30) ──")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X, y))

    # clf_cm è un modello temporaneo addestrato sul 70%: serve solo per la CM.
    # Viene scartato al termine di questa sezione.
    clf_cm = CLASSIFIER_FACTORY[clf_name]()
    clf_cm.fit(X[train_idx], y[train_idx])
    y_pred = clf_cm.predict(X[test_idx])

    # confusion_matrix con labels=class_names forza l'ordine delle righe/colonne:
    # senza questo argomento sklearn usa l'ordine dei valori unici in y,
    # che potrebbe differire dall'ordine alfabetico atteso nella heatmap.
    cm       = confusion_matrix(y[test_idx], y_pred, labels=class_names)
    cm_title = f"Confusion Matrix — {desc_type} K={k} {clf_name}"
    cm_path  = RESULTS_DIR / "confusion_matrix.png"

    plot_confusion_matrix(cm, class_names, title=cm_title, save_path=cm_path)

    # Accuracy sullo split 30%: indicativa, non è la metrica ufficiale.
    # Tipicamente leggermente superiore alla CV perché il train set è più grande
    # (70% vs ~66% per fold in 3-fold CV).
    acc_split = (y_pred == y[test_idx]).mean()
    print(f"Accuracy sullo split 30% : {acc_split:.3f}")

    # classification_report mostra precision, recall e F1 per singola classe:
    # utile per identificare le classi problematiche (es. sparseresidential)
    # e confrontarle con la matrice di confusione visuale.
    report = classification_report(y[test_idx], y_pred, target_names=class_names)
    print("\n── Classification Report (split 30%) ──")
    print(report)
    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[saved] {report_path}")

    # ── 2. Training finale su tutto il dataset ────────────────────────────────
    # Il modello di deployment viene addestrato su TUTTI i 2100 campioni UCMerced.
    # Più dati di training → confini decisionali più stabili → migliore
    # generalizzazione su immagini nuove in inferenza.
    # La metrica di riferimento rimane quella della CV (cv_results.csv).
    print("\n── Training finale su UC Merced completo ──")
    clf_final = CLASSIFIER_FACTORY[clf_name]()
    with Timer():
        clf_final.fit(X, y)

    save_pickle(clf_final, FINAL_MODEL_FILE)

    # ── 3. Copia vocabolario vincente come "vocabulary_final" ─────────────────
    # Rinominiamo il vocabolario vincente in un nome stabile (vocabulary_final.pkl)
    # così inference.py non deve sapere quale K o desc_type ha vinto: carica
    # sempre lo stesso file. shutil.copy preserva il contenuto binario del pickle
    # senza ricaricare o risalvare l'oggetto KMeans, evitando overhead.
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
