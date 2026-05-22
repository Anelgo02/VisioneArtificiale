"""
Parametri centralizzati per la pipeline Bag-of-Visual-Words.
Modificare qui per cambiare percorsi, K o altri iperparametri.
"""

from pathlib import Path

# ── Riproducibilità ──────────────────────────────────────────────────────────
# Seme fisso usato da numpy, sklearn e KMeans in tutta la pipeline.
# Garantisce che rieseguendo gli script su macchine diverse si ottengano
# gli stessi artefatti (centroidi, fold, split, modelli).
RANDOM_SEED = 42

# ── Cartelle di output ───────────────────────────────────────────────────────
# Path(__file__).parent risolve la cartella dello script stesso, rendendo
# tutti i percorsi relativi indipendenti dalla working directory corrente.
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

# Le cartelle vengono create al momento dell'import di config:
# ogni script può assumere che models/ e results/ esistano già.
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Percorsi dataset ──────────────────────────────────────────────────────────
#   Struttura attesa:
#     datasets/AID/<classe>/<immagine>.jpg
#     datasets/UCMerced/Images/<classe>/<immagine>.tif
#   I dataset NON sono in git: vanno copiati manualmente sulla macchina di lavoro.
DATASETS_DIR = PROJECT_ROOT / "datasets"
AID_ROOT     = DATASETS_DIR / "AID"
UCM_ROOT     = DATASETS_DIR / "UCMerced" / "Images"

# ── Descrittori locali ───────────────────────────────────────────────────────
# "SIFT" obbligatorio; aggiungere "ORB" per il confronto opzionale.
# Tutti gli script iterano su questa lista: aggiungere "ORB" qui è sufficiente
# per abilitarlo in tutta la pipeline senza altre modifiche.
DESCRIPTOR_TYPES = ["SIFT"]       # esempio con entrambi: ["SIFT", "ORB"]

# Numero massimo di descrittori tenuti per immagine AID durante l'estrazione.
# Bilancia diversità del vocabolario (più descrittori = più varietà) e uso RAM
# (10.000 img AID × 100 desc × 128-d float32 ≈ 512 MB).
MAX_DESCRIPTORS_PER_IMAGE = 100

# ── Estrazione descrittori (Fase 1a) ─────────────────────────────────────────
def descriptors_file(desc_type: str) -> Path:
    """Percorso del file pickle con tutti i descrittori AID per desc_type."""
    return MODELS_DIR / f"descriptors_aid_{desc_type}.pkl"

# ── Costruzione vocabolario (Fase 1b) ────────────────────────────────────────
# Valori di K da testare come asse sperimentale richiesto dalla consegna.
# K=50  → vocabolario grossolano, vettori BoW molto compatti
# K=100 → compromesso intermedio
# K=500 → vocabolario fine, maggiore potere discriminativo
K_VALUES = [50, 100, 500]

def vocabulary_file(desc_type: str, k: int) -> Path:
    """Percorso del modello KMeans (vocabolario) per la combinazione (desc_type, K)."""
    return MODELS_DIR / f"vocabulary_{desc_type}_K{k}.pkl"

# ── Rappresentazione BoW (Fase 2) ────────────────────────────────────────────
def bow_file(desc_type: str, k: int) -> Path:
    """Percorso del dizionario {X, y, class_names} per la combinazione (desc_type, K)."""
    return MODELS_DIR / f"bow_ucmerced_{desc_type}_K{k}.pkl"

# ── Cross-validation (Fase 3) ────────────────────────────────────────────────
# 3-fold è il minimo che garantisce un test set di dimensione ragionevole
# (33% = ~693 campioni su 2100) mantenendo tempi di training accettabili.
N_FOLDS         = 3
CV_RESULTS_FILE = RESULTS_DIR / "cv_results.csv"

# ── Training finale (Fase 4) ─────────────────────────────────────────────────
# Nome stabile degli artefatti di deployment: inference.py carica sempre questi
# due file senza dover sapere quale (desc_type, K) ha vinto la CV.
FINAL_MODEL_FILE  = MODELS_DIR / "classifier_final.pkl"
FINAL_VOCAB_FILE  = MODELS_DIR / "vocabulary_final.pkl"

# ── MiniBatchKMeans ───────────────────────────────────────────────────────────
# KMEANS_BATCH_SIZE: campioni per mini-batch. 10.000 su ~1M descrittori è
#   circa l'1% del dataset per iterazione: buon compromesso velocità/stabilità.
# KMEANS_MAX_ITER: iterazioni massime prima dello stop forzato.
#   In pratica MiniBatchKMeans converge prima grazie all'early stopping interno.
#   Se n_iter_ == KMEANS_MAX_ITER nell'output, aumentare questo valore.
KMEANS_BATCH_SIZE = 10_000
KMEANS_MAX_ITER   = 300
