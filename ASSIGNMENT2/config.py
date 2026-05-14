"""
Parametri centralizzati per la pipeline Bag-of-Visual-Words.
Modificare qui per cambiare percorsi, K o altri iperparametri.
"""

from pathlib import Path

# ── Riproducibilità ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Cartelle di output ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
# ── Percorsi dataset ──────────────────────────────────────────────────────────
#   Struttura attesa:
#     datasets/AID/<classe>/<immagine>.jpg
#     datasets/UCMerced/Images/<classe>/<immagine>.tif
DATASETS_DIR = PROJECT_ROOT / "datasets"
AID_ROOT     = DATASETS_DIR / "AID"
UCM_ROOT     = DATASETS_DIR / "UCMerced" / "Images"

# ── Estrazione descrittori (Fase 1a) ─────────────────────────────────────────
MAX_DESCRIPTORS_PER_IMAGE = 100   # cap per immagine AID
DESCRIPTORS_FILE = MODELS_DIR / "descriptors_aid.pkl"

# ── Costruzione vocabolario (Fase 1b) ────────────────────────────────────────
K_VALUES = [50, 100, 500]         # dimensioni del vocabolario da testare

def vocabulary_file(k: int) -> Path:
    return MODELS_DIR / f"vocabulary_K{k}.pkl"

# ── Rappresentazione BoW (Fase 2) ────────────────────────────────────────────
def bow_file(k: int) -> Path:
    return MODELS_DIR / f"bow_ucmerced_K{k}.pkl"

# ── Cross-validation (Fase 3) ────────────────────────────────────────────────
N_FOLDS = 3
CV_RESULTS_FILE = RESULTS_DIR / "cv_results.csv"

# ── Training finale (Fase 4) ─────────────────────────────────────────────────
FINAL_MODEL_FILE  = MODELS_DIR / "classifier_final.pkl"
FINAL_VOCAB_FILE  = MODELS_DIR / "vocabulary_final.pkl"

# ── MiniBatchKMeans ───────────────────────────────────────────────────────────
KMEANS_BATCH_SIZE = 10_000
KMEANS_MAX_ITER   = 300
