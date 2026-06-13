
# Assignment 3 — Classificazione di Immagini Aeree con CNN Residuale

Corso di Visione Artificiale (LM-32). 
 
Classificazione di immagini aeree su **21 classi** (dataset UC Merced) tramite una CNN residuale custom con transfer learning da AID.

## Struttura del progetto

```
ASSIGNMENT3/
├── config.py       — tutti gli iperparametri (modificare solo questo file)
├── model.py        — blocco residuale custom + architettura CNN + replace_head
├── dataset.py      — caricamento OpenCV, split stratificato, augmentation
├── train.py        — Strategia 1 (pretrain AID + finetune UC) e Strategia 2 (scratch)
├── inference.py    — predizione su singola immagine
├── evaluate.py     — valutazione del best model sul test set (classification report)
├── data/
│   ├── AID/                        — dataset pre-training (30 classi, ~10k immagini)
│   └── UCMerced_LandUse/Images/    — dataset target (21 classi, 2100 immagini)
├── models/         — modelli salvati dopo il training (.keras)
└── results/        — curve di training e confusion matrix (.png)
```

## Installazione

### Con conda (consigliato)
```bash
conda create -n visioneArtificiale python=3.10
conda activate visioneArtificiale
pip install -r requirements.txt
```

### Con pip e virtualenv
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

### Senza ambiente virtuale
```bash
pip install -r requirements.txt
```

## Utilizzo

### Training completo
```bash
conda activate visioneArtificiale
python train.py
```

Esegue in sequenza:
1. Pre-training su AID (30 classi)
2. Fine-tuning su UC Merced (21 classi)
3. Training da zero su UC Merced
4. Model selection sul validation set (F1-macro)
5. Valutazione finale sul test set (Accuracy, Precision, Recall, F1, Confusion Matrix)

### Valutazione del modello già addestrato
```bash
python evaluate.py
```

Carica il modello salvato in `models/best_model.keras` e stampa il classification report completo sul test set, senza rieseguire il training.

### Test rapido (poche epoche per verificare che non ci siano errori)
Modifica `config.py` impostando `PRETRAIN_EPOCHS = 3`, `FINETUNE_EPOCHS = 3`, `SCRATCH_EPOCHS = 3`, poi:
```bash
python train.py
```

### Inferenza su una singola immagine
```bash
python inference.py path/alla/immagine.jpg
python inference.py path/alla/immagine.jpg models/best_model.keras
```

## Requisiti tecnici rispettati

- Framework: TensorFlow/Keras
- Manipolazione immagini: OpenCV
- Loss: categorical cross-entropy
- Architettura: CNN residuale custom con 3 blocchi residuali
- Flatten prima del classificatore (no average pooling)
- MLP finale: 2 fully connected layers (256 → 128)
- Early Stopping in tutti gli esperimenti
- Split UC Merced: 70/15/15 fisso e stratificato
- Data augmentation solo sul training set

## Iperparametri principali

| Parametro | Valore |
|---|---|
| Dimensione immagini | 128×128 |
| Filtri stages | 32 → 64 → 128 |
| MLP units | [256, 128] |
| Dropout | 0.4 |
| LR pre-training | 1e-3 |
| LR fine-tuning | 1e-4 |
| Strategia fine-tuning | partial (60% layer congelati) |
| Batch size | 32 |
| Seed split | 42 |

---

Dataset Download (AID) : https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets?resource=download
Dataset Download (UC Merced Land use) : https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset