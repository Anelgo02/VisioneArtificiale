# CLAUDE.md — Configurazione Claude Code per Assignment 3

## Contesto del progetto

Assignment universitario per il corso di Visione Artificiale (LM-32).
Obiettivo: classificazione di immagini aeree (21 classi, dataset UC Merced)
tramite CNN residuale custom + transfer learning da AID.

## Struttura del progetto

```
assignment3/
├── CLAUDE.md          ← questo file
├── config.py          ← TUTTI gli iperparametri qui, mai hardcodati altrove
├── model.py           ← blocco residuale custom + architettura CNN
├── dataset.py         ← caricamento OpenCV, split stratificato, augmentation
├── train.py           ← Strategia 1 (pretrain+finetune) e Strategia 2 (scratch)
├── inference.py       ← predizione su singola immagine
├── data/
│   ├── AID/           ← dataset pre-training (30 classi, ~10k immagini)
│   └── UCMerced_LandUse/Images/  ← dataset target (21 classi, ~2100 immagini)
├── models/            ← modelli salvati (.keras)
└── results/           ← curve di training, confusion matrix (PNG)
```

## Vincoli tecnici obbligatori (dal testo dell'assignment)

- Framework: **TensorFlow/Keras** esclusivamente
- Manipolazione immagini: **OpenCV** esclusivamente (no PIL, no tf.image per load)
- Loss: **categorical cross-entropy**
- NO average pooling prima del classificatore finale
- NO ResNet predefinita — il blocco residuale deve essere implementato custom
- Almeno **2 blocchi residuali**
- **Flatten** prima del classificatore (non GlobalAveragePooling)
- MLP finale: 1-3 fully connected layers
- **Early Stopping** in tutti gli esperimenti
- Split UC Merced: **70/15/15 fisso e stratificato** — fatto una sola volta
- Data augmentation **solo sul training set** (no data leakage)
- Immagini ridimensionate sotto 200x200 (usiamo 128x128)

## Iperparametri principali (tutti in config.py)

| Parametro | Valore | Motivazione |
|---|---|---|
| IMG_SIZE | 128x128 | Compromesso velocità/qualità su CPU |
| PRETRAIN_LR | 1e-3 | Adam standard per training da zero |
| FINETUNE_LR | 1e-4 | 10x più basso per non distruggere i pesi |
| DROPOUT_RATE | 0.4 | Regolarizzazione MLP finale |
| FINETUNE_STRATEGY | "partial" | Da motivare nella presentazione |

## Regole di sviluppo

### Modifica dei parametri
Modifica SEMPRE e SOLO config.py per cambiare iperparametri.
Non hardcodare mai numeri in model.py, dataset.py o train.py.

### Aggiunta di nuove funzionalità
- Nuova metrica → aggiungi in train.py nella funzione `evaluate_on_test()`
- Nuova tecnica di augmentation → aggiungi in dataset.py nella funzione `augment_image()`
- Variante architetturale → crea una nuova funzione in model.py, non modificare `build_model()`

### Salvataggio risultati
- Modelli: sempre in `models/` con estensione `.keras`
- Plot: sempre in `results/` come PNG a 150 dpi
- Non salvare mai file temporanei nella root del progetto

## Come eseguire

```bash
# Installazione dipendenze
pip install tensorflow opencv-python scikit-learn matplotlib

# Test rapido (poche epoche per verificare che non ci siano errori)
# Prima modifica config.py: PRETRAIN_EPOCHS=3, SCRATCH_EPOCHS=3
python train.py

# Training completo
python train.py

# Inferenza su immagine singola
python inference.py path/alla/immagine.jpg

# Inferenza con modello specifico
python inference.py path/alla/immagine.jpg models/model_finetuned_UC.keras
```

## Note per l'esame orale

La professoressa chiederà di giustificare ogni scelta. Argomenti chiave:

**Sul blocco residuale:**
- Perché la skip connection risolve il vanishing gradient
- Formula: y = F(x) + x, derivata: ∂y/∂x = ∂F(x)/∂x + I
- Perché serve la proiezione 1x1 quando i canali cambiano

**Sul fine-tuning:**
- Perché il LR deve essere più basso (non distruggere i pesi pre-addestrati)
- Differenza tra frozen / partial / full e quando conviene ciascuno
- Quanti parametri sono addestrabili in ogni strategia (model.py li stampa)

**Sul dataset split:**
- Perché stratificato (preserva proporzione classi in ogni split)
- Perché il test set viene toccato una sola volta alla fine
- Perché l'augmentation non va applicata a val e test (data leakage)

**Sulla loss:**
- Categorical cross-entropy: formula, perché softmax + CE è efficiente
- Differenza tra sparse_categorical_crossentropy e categorical_crossentropy

## Ambiente Python consigliato

```bash
conda create -n cv_assignment3 python=3.11
conda activate cv_assignment3
pip install tensorflow opencv-python scikit-learn matplotlib
```

## Checklist pre-consegna

- [ ] Il codice gira senza errori da `python train.py`
- [ ] I plot delle curve vengono salvati in `results/`
- [ ] La confusion matrix viene salvata in `results/`
- [ ] `python inference.py <immagine>` stampa la classe predetta
- [ ] Il report PDF descrive la strategia di fine-tuning scelta e la motiva
- [ ] Il numero di parametri addestrabili per ogni strategia è riportato nelle slide
- [ ] Early stopping è attivo in tutti gli esperimenti
- [ ] Lo split UC Merced è fisso (stesso seed) per entrambe le strategie
