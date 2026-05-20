# Assignment 2 — Classificazione di Immagini Aeree con Bag-of-Visual-Words

## Descrizione

Classificatore per immagini aeree RGB su 21 classi del dataset **UC Merced**, usando l'approccio classico Bag-of-Visual-Words:

1. Vocabolario visuale costruito con descrittori SIFT estratti da **AID** (etichette ignorate)
2. Rappresentazione di ogni immagine come istogramma di visual words (normalizzato L2)
3. Confronto tra SVM-RBF e Random Forest con 3-fold StratifiedKFold
4. Modello migliore: **SIFT + K=500 + SVM-RBF** → accuracy **71.6%**, F1 macro **71.3%**

---

## Requisiti

```bash
conda create -n visioneArtificiale python=3.11 -y
conda activate visioneArtificiale
pip install -r requirements.txt
```

---

## Dataset

Scaricare e posizionare i dataset nella cartella `datasets/` con questa struttura:

```
datasets/
├── AID/
│   ├── Airport/
│   ├── BareLand/
│   └── ...          (30 classi, ~10.000 immagini .jpg)
└── UCMerced/
    └── Images/
        ├── agricultural/
        ├── airplane/
        └── ...      (21 classi, 100 immagini .tif per classe)
```

- **AID**: [Xia et al., IEEE TGRS 2017]
- **UC Merced**: [Yang & Newsam, ACM GIS 2010]

---

## Esecuzione della pipeline

Gli script vanno eseguiti in ordine. Ogni script è **idempotente**: se l'artefatto di output esiste già, salta il ricalcolo.

```bash
conda activate visioneArtificiale
cd ASSIGNMENT2/

# Fase 0 — Verifica integrità dataset
python 00_check_datasets.py

# Fase 1a — Estrazione descrittori SIFT da AID (~15-20 min)
python 01_extract_descriptors.py

# Fase 1b — Costruzione vocabolario K ∈ {50, 100, 500} (~10 min)
python 02_build_vocabulary.py

# Fase 2 — Istogrammi BoW per UC Merced (~5 min)
python 03_compute_bow.py

# Fase 3 — Cross-validation: SVM-RBF vs Random Forest (~30 min)
python 04_cross_validation.py

# Fase 4 — Training finale + matrice di confusione (~5 min)
python 05_train_final.py
```

---

## Inferenza su una singola immagine

```bash
python inference.py path/alla/immagine.jpg
```

Output esempio:
```
Modello caricato  : classifier_final.pkl
Vocabolario       : vocabulary_final.pkl  (K=500)
Descrittore       : SIFT
Immagine          : test.jpg

Classe predetta   : forest
```

---

## Struttura del progetto

```
ASSIGNMENT2/
├── config.py                  # parametri centralizzati (percorsi, K, seed)
├── utils.py                   # funzioni helper condivise
├── requirements.txt
│
├── 00_check_datasets.py       # verifica dataset prima di partire
├── 01_extract_descriptors.py  # SIFT su AID → models/descriptors_aid_SIFT.pkl
├── 02_build_vocabulary.py     # K-Means → models/vocabulary_SIFT_K{k}.pkl
├── 03_compute_bow.py          # istogrammi BoW → models/bow_ucmerced_SIFT_K{k}.pkl
├── 04_cross_validation.py     # 3-fold CV → results/cv_results.csv
├── 05_train_final.py          # training finale → models/classifier_final.pkl
├── inference.py               # classificazione singola immagine
│
├── models/                    # artefatti generati (non inclusi nel repository)
└── results/                   # cv_results.csv, confusion_matrix.png
```

---

## Risultati

| K | Classificatore | Accuracy | F1 macro |
|---|----------------|----------|----------|
| 50 | SVM-RBF | 66.6% ± 1.4% | 66.2% ± 1.5% |
| 50 | Random Forest | 62.5% ± 0.2% | 61.2% ± 0.3% |
| 100 | SVM-RBF | 68.2% ± 1.1% | 67.8% ± 1.1% |
| 100 | Random Forest | 61.8% ± 0.5% | 60.5% ± 0.6% |
| **500** | **SVM-RBF** | **71.6% ± 1.2%** | **71.3% ± 1.0%** |
| 500 | Random Forest | 60.1% ± 0.5% | 58.2% ± 0.7% |

La matrice di confusione del modello vincente è in `results/confusion_matrix.png`.
