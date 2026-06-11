

# ── Dataset paths ─────────────────────────────────────────────────────────────
AID_DIR   = "data/AID"          
UC_DIR    = "data/UCMerced_LandUse/Images"

# ── Immagini ───────────────────────────────────────────────────────────────────
IMG_SIZE  = (128, 128)          # sotto 200x200; 128 è un buon compromesso velocità/qualità su hardware consumer
CHANNELS  = 3                   # RGB

# ── Split UC Merced ────────────────────────────────────────────────────────────
# Il task richiede partizione FISSA: 70/15/15 stratificata
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42                # per riproducibilità

# ── Architettura ───────────────────────────────────────────────────────────────

FILTERS_STAGE_1 = 32            # primo stage convoluzionale
FILTERS_STAGE_2 = 64            # secondo stage
FILTERS_STAGE_3 = 128           # terzo stage 
KERNEL_SIZE     = 3             # 3x3: cattura pattern locali, standard in letteratura
DROPOUT_RATE    = 0.4           # regolarizzazione nel MLP finale
# MLP finale
MLP_UNITS       = [256, 128]    # due FC layers prima dell'output

# ── Classi ────────────────────────────────────────────────────────────────────
AID_CLASSES = 30                # classi del dataset AID (pre-training)
UC_CLASSES  = 21                # classi UC Merced (fine-tuning e training da zero)

UC_CLASS_NAMES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark",
    "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt"
]

# ── Training - Pre-training su AID ────────────────────────────────────────────
PRETRAIN_EPOCHS    = 50
PRETRAIN_LR        = 1e-3       # Adam standard per training da zero
PRETRAIN_BATCH     = 32
PRETRAIN_PATIENCE  = 8          # early stopping: ferma se val_loss non migliora per N epoche

# ── Training - Fine-tuning su UC ──────────────────────────────────────────────
# Il LR deve essere PIÙ BASSO del pre-training: non vogliamo distruggere
# i pesi già appresi, ma solo adattarli. Tipicamente 10x-100x più piccolo.
FINETUNE_EPOCHS   = 50
FINETUNE_LR       = 1e-4        # un ordine di grandezza sotto PRETRAIN_LR
FINETUNE_BATCH    = 32
FINETUNE_PATIENCE = 8

# Strategie di fine-tuning: si è scelto di optare per la la strategia partial
# "frozen"   -> congela tutto il backbone, addestra solo MLP
# "partial"  -> congela i primi N layer, lascia liberi gli ultimi convoluzionali
# "full"     -> addestra tutto il modello
FINETUNE_STRATEGY = "partial"   

# ── Training - da zero su UC (Strategia 2) ────────────────────────────────────
SCRATCH_EPOCHS   = 80
SCRATCH_LR       = 1e-3
SCRATCH_PATIENCE = 10

# ── Paths di salvataggio ──────────────────────────────────────────────────────
PRETRAINED_MODEL_PATH  = "models/model_pretrained_AID.keras"
FINETUNED_MODEL_PATH   = "models/model_finetuned_UC.keras"
SCRATCH_MODEL_PATH     = "models/model_scratch_UC.keras"
BEST_MODEL_PATH        = "models/best_model.keras"  # quello selezionato sul val set
