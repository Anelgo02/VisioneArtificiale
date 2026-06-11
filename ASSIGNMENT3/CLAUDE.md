# CLAUDE.md — Assignment 3 · Visione Artificiale (LM-32)

## Contesto del progetto

Assignment universitario per il corso di Visione Artificiale (LM-32).
Obiettivo: classificazione di immagini aeree (21 classi, dataset UC Merced)
tramite CNN residuale custom + transfer learning da AID.
Consegna: **14 giugno 2026**.

## Struttura del progetto

```
ASSIGNMENT3/
├── CLAUDE.md          ← questo file
├── config.py          ← TUTTI gli iperparametri qui, mai hardcodati altrove
├── model.py           ← blocco residuale custom + architettura CNN
├── dataset.py         ← caricamento OpenCV, split stratificato, augmentation
├── train.py           ← Strategia 1 (pretrain+finetune) e Strategia 2 (scratch)
├── inference.py       ← predizione su singola immagine
├── data/
│   ├── AID/                       ← dataset pre-training (30 classi, ~10k immagini)
│   └── UCMerced_LandUse/Images/   ← dataset target (21 classi, 2100 immagini)
├── models/            ← modelli salvati (.keras)
└── results/           ← curve di training e confusion matrix (PNG)
```

---

## Architettura del modello (da model.py)

### Blocco residuale custom
Formula: **y = F(x) + x**
- Ramo principale: Conv2D 3×3 → BN → ReLU → Conv2D 3×3 → BN
- Skip connection: identità se i canali coincidono; Conv2D 1×1 + BN se cambiano (proiezione)
- ReLU finale dopo la somma
- Nessun bias nelle conv (BN lo gestisce implicitamente)

### Architettura completa (build_model)
```
Input (128×128×3)
→ Stem: Conv2D 3×3 (32 filtri) → BN → ReLU
→ Stage 1: ResidualBlock(32) → MaxPool 2×2   [128→64]
→ Stage 2: ResidualBlock(64) → MaxPool 2×2   [64→32]
→ Stage 3: ResidualBlock(128) → MaxPool 2×2  [32→16]
→ Flatten  [16×16×128 = 32768 neuroni]
→ Dense(256) → ReLU → Dropout(0.4)
→ Dense(128) → ReLU → Dropout(0.4)
→ Dense(num_classes) → Softmax
```

### replace_head (per fine-tuning)
Prende il modello pre-addestrato su AID (30 classi), rimuove l'ultimo Dense,
aggiunge Dense(21) + Softmax. Strategie disponibili:
- **frozen**: congela tutto il backbone, addestra solo il nuovo Dense(21)
- **partial**: congela il 60% iniziale dei layer, lascia liberi gli ultimi conv + MLP
- **full**: tutto addestrabile, LR basso per non distruggere i pesi

Strategia usata: **partial** (motivazione: compromesso tra trasferimento delle feature
di basso livello e adattamento delle feature di alto livello al nuovo dominio).

---

## Iperparametri completi (config.py)

| Parametro | Valore | Motivazione |
|---|---|---|
| IMG_SIZE | 128×128 | < 200×200 come richiesto; compromesso velocità/qualità |
| CHANNELS | 3 | RGB |
| FILTERS_STAGE_1 | 32 | base |
| FILTERS_STAGE_2 | 64 | raddoppio: meno spazio, più semantica |
| FILTERS_STAGE_3 | 128 | idem |
| KERNEL_SIZE | 3×3 | pattern locali, standard ResNet |
| DROPOUT_RATE | 0.4 | regolarizzazione MLP |
| MLP_UNITS | [256, 128] | 2 FC layers |
| PRETRAIN_EPOCHS | 50 | con early stopping (patience=8) |
| PRETRAIN_LR | 1e-3 | Adam standard per training da zero |
| PRETRAIN_BATCH | 32 | buon equilibrio velocità/stabilità |
| FINETUNE_EPOCHS | 50 | con early stopping (patience=8) |
| FINETUNE_LR | 1e-4 | 10× più basso: non distrugge i pesi appresi |
| FINETUNE_BATCH | 32 | |
| FINETUNE_STRATEGY | "partial" | 60% layer congelati |
| SCRATCH_EPOCHS | 80 | con early stopping (patience=10) |
| SCRATCH_LR | 1e-3 | |
| RANDOM_SEED | 42 | riproducibilità split |
| TRAIN/VAL/TEST | 70/15/15 | stratificato, fisso per entrambe le strategie |

---

## Dataset

### AID (pre-training)
- ~10.000 immagini, 30 classi di scene aeree
- Split usato: 70% train / 30% val (stratificato)
- Nessun test set per AID (non serve: è solo pre-training)

### UC Merced Land Use (target)
- 2.100 immagini, 21 classi, 100 immagini per classe
- Split fisso (seed=42): **1470 train / 315 val / 315 test**
- Stesso split usato per entrambe le strategie
- 21 classi: agricultural, airplane, baseballdiamond, beach, buildings, chaparral,
  denseresidential, forest, freeway, golfcourse, harbor, intersection,
  mediumresidential, mobilehomepark, overpass, parkinglot, river, runway,
  sparseresidential, storagetanks, tenniscourt

### Preprocessing (identico in training e inferenza)
- OpenCV: lettura BGR → conversione RGB
- Resize a 128×128 (cv2.resize, interpolazione default)
- Normalizzazione [0, 1]: divisione per 255.0
- Tipo: float32

### Data augmentation (solo training set)
- Random flip orizzontale e verticale
- Random brightness (max_delta=0.15)
- Random contrast (lower=0.85, upper=1.15)
- Clip [0, 1] dopo augmentation

---

## Risultati ottenuti (cartella results/)

### Curva pre-training AID (pretrain_AID_curves.png)
- 50 epoche completate (early stopping non scattato → ancora in apprendimento)
- Loss: train scende da ~4.5 a ~1.7; val da ~3.4 a ~1.5
- Accuracy: train sale da ~5% a ~47%; val da ~5% a ~53%
- Val accuracy > train accuracy: sano, nessun overfitting (Dropout attivo solo in train)
- Il modello ha appreso feature generalizzabili su 30 classi AID

### Curva fine-tuning UC partial (finetune_UC_curves.png)
- 50 epoche completate (ancora in miglioramento alla fine)
- Loss: train scende da ~3.6 a ~1.7; val da ~3.1 a ~1.3
- Accuracy: train sale da ~6% a ~44%; val da ~6% a ~67%
- Val accuracy significativamente superiore a train (effetto Dropout)
- Convergenza lenta ma costante: il transfer learning ha funzionato

### Curva training da zero UC (scratch_UC_curves.png)
- Solo ~11 epoche prima dell'early stopping (patience=10)
- Loss: train crolla da ~7.5 a ~3.0 in 1 epoca; val piatta a ~3.0 dall'inizio
- Accuracy: train ~3.5–5.7%; val ~4.7% (praticamente casuale su 21 classi)
- **La rete non ha imparato nulla di utile**: con soli 1470 campioni di training
  e nessun pre-training, la rete residuale custom non riesce a convergere

### Model selection
- Strategia 1 (fine-tuning partial): val accuracy ~67%, val loss ~1.3
- Strategia 2 (scratch): val accuracy ~5%, val loss ~3.0
- **Selezionata Strategia 1** (F1-macro nettamente superiore su val set)

### Confusion matrix test set (confusion_matrix_test.png)
Il modello selezionato è quello della Strategia 1 (fine-tuning partial).
Classi classificate bene (diagonale alta):
- baseballdiamond: 15/~15
- chaparral: 15/~15
- beach: 14/~15
- mobilehomepark: 14/~15
- parkinglot: 14/~15
- airplane: 12/~15
- mediumresidential: 11/~15
- overpass: 11/~15

Classi con più errori / confusioni frequenti:
- forest ↔ chaparral (texture simile: vegetazione densa)
- river ↔ sparseresidential (forme allungate)
- tenniscourt → confuso con più classi
- buildings ↔ denseresidential / mediumresidential
- freeway ↔ runway (strutture lineari)

Accuracy complessiva sul test set: stimata ~55–65% (leggibile dalla diagonale).

---

## Vincoli tecnici rispettati

- TensorFlow/Keras esclusivamente
- OpenCV per tutte le manipolazioni immagini (no PIL, no tf.image per load)
- Categorical cross-entropy
- NO average pooling prima del classificatore
- NO ResNet predefinita: blocco residuale implementato custom
- 3 blocchi residuali (≥2 richiesti)
- Flatten prima del MLP
- MLP: 2 FC layers (256 → 128)
- Early Stopping in tutti gli esperimenti
- Split 70/15/15 fisso e stratificato (seed=42), fatto una sola volta
- Augmentation solo su training set

---

## SEZIONE PRESENTAZIONE

### Contesto
L'utente deve produrre una presentazione in formato slide (PDF) da consegnare
insieme al codice entro il 14 giugno 2026.

### Requisiti formali
- Massimo **14 slide** (esclusa la slide aggiuntiva finale)
- Lingua: italiana
- Tono: universitario, chiaro e sintetico
- L'utente stava già scrivendo il primo capitolo quando ha richiesto assistenza

### Struttura richiesta (14 slide + 1)

**Slide 1 — Titolo / Introduzione**
- Titolo: Assignment 3 – Classificazione di Immagini Aeree con CNN Residuale
- Corso, autore, data
- Breve frase sull'obiettivo: classificare immagini aeree in 21 categorie

**Slide 2 — Pipeline del metodo**
- Diagramma/schema delle due strategie (come nel PDF dell'assignment)
- Strategia 1: AID pretrain → replace head → UC fine-tune
- Strategia 2: UC train from scratch
- Confronto finale su val set → test set

**Slide 3 — Dataset**
- AID: ~10k immagini, 30 classi, split 70/30
- UC Merced: 2100 immagini, 21 classi, 100 per classe, split 70/15/15 fisso stratificato
- Preprocessing: resize 128×128, normalizzazione [0,1], OpenCV BGR→RGB
- Augmentation: flip, brightness, contrast (solo training)

**Slide 4 — Architettura: blocco residuale**
- Schema del blocco: Conv→BN→ReLU→Conv→BN + skip connection
- Formula y = F(x) + x
- Motivazione: risolve vanishing gradient (∂y/∂x = ∂F/∂x + I)
- Proiezione 1×1 quando i canali cambiano

**Slide 5 — Architettura: rete completa**
- Tabella o schema: Input → Stem → Stage1(32) → Pool → Stage2(64) → Pool → Stage3(128) → Pool → Flatten → Dense(256) → Dropout → Dense(128) → Dropout → Dense(21) → Softmax
- Numero totale parametri (da calcolare/stampare con model.summary())
- Giustificazione Flatten (no average pooling: requisito)

**Slide 6 — Iperparametri principali**
- Tabella compatta con i valori e una breve motivazione per ciascuno
- Evidenziare: LR pre-training vs LR fine-tuning (10× differenza)
- Dropout 0.4, batch 32, seed 42

**Slide 7 — Strategia 1: Pre-training su AID**
- Curva loss e accuracy (pretrain_AID_curves.png)
- Commento: convergenza sana, val > train (no overfitting)
- 50 epoche, accuracy ~53% su 30 classi con rete piccola custom

**Slide 8 — Strategia 1: Fine-tuning su UC Merced**
- Curva loss e accuracy (finetune_UC_curves.png)
- Strategia "partial": 60% layer congelati
- Commento: val accuracy ~67%, convergenza lenta ma costante
- Perché LR basso (1e-4): non distruggere i pesi appresi su AID

**Slide 9 — Strategia 2: Training da zero**
- Curva (scratch_UC_curves.png)
- Early stopping dopo ~11 epoche
- Commento: fallimento completo (~5% accuracy ≈ caso su 21 classi)
- Motivazione: 1470 immagini insufficienti per addestrare una CNN da zero

**Slide 10 — Numero di parametri addestrabili (fine-tuning)**
- Tabella con le tre strategie:
  | Strategia | Param. addestrabili | Param. congelati |
  | frozen    | solo Dense(21)      | tutto il backbone |
  | partial   | ~40% della rete     | ~60% iniziale |
  | full      | tutti               | nessuno |
- I valori esatti vengono stampati da replace_head() in model.py

**Slide 11 — Model Selection**
- Confronto val F1-macro: Strategia 1 >> Strategia 2
- Grafico o tabella comparativa
- Motivazione della scelta: transfer learning essenziale con dataset piccolo
- Il test set viene toccato UNA SOLA VOLTA sul modello selezionato

**Slide 12 — Risultati sul test set**
- Tabella: Accuracy, Precision, Recall, F1-macro (macro-average)
- Valori stimati dal confusion matrix (~55–65% accuracy)
- Classification report per classe (se spazio)

**Slide 13 — Confusion Matrix e risultati qualitativi**
- Immagine confusion_matrix_test.png
- Classi facili: baseballdiamond, chaparral, beach, parkinglot (struttura visiva distintiva)
- Classi difficili / confusioni tipiche:
  - forest ↔ chaparral: texture vegetazione simile
  - river ↔ sparseresidential: forme allungate
  - buildings ↔ denseresidential: difficile separare densità residenziale
  - freeway ↔ runway: strutture lineari simili

**Slide 14 — Possibili miglioramenti**
- Più epoche / LR scheduling (cosine annealing)
- Architettura più profonda o uso di backbone pretrained più potente (EfficientNet, ResNet50)
- Augmentation più aggressiva (rotazioni, crop)
- Strategia full fine-tuning con warm-up progressivo del LR
- Ensemble dei due modelli
- Test Time Augmentation (TTA)

**Slide 15 (aggiuntiva obbligatoria) — Riferimenti e AI**
- Riferimenti: He et al. 2016 (Deep Residual Learning), dataset AID, dataset UC Merced
- Risorse utilizzate: TensorFlow docs, scikit-learn docs
- Uso di AI: Claude Code (Anthropic) usato per debugging, revisione codice,
  comprensione teorica dei concetti (vanishing gradient, transfer learning,
  strategie di fine-tuning)

---

## Regole di sviluppo

- Modifica SEMPRE e SOLO config.py per cambiare iperparametri
- Nuova metrica → aggiungi in train.py in evaluate_on_test()
- Nuova augmentation → aggiungi in dataset.py in augment_image()
- Variante architetturale → nuova funzione in model.py, non modificare build_model()
- Modelli: sempre in models/ con estensione .keras
- Plot: sempre in results/ come PNG a 150 dpi

## Come eseguire

```bash
conda activate visioneArtificiale
python train.py          # training completo
python inference.py path/immagine.jpg   # inferenza singola immagine
```

## Checklist pre-consegna

- [x] Codice funzionante (risultati già generati in results/)
- [x] Curve di training salvate in results/
- [x] Confusion matrix salvata in results/
- [x] inference.py funzionante
- [ ] Presentazione PDF completata (max 14 slide + 1)
- [ ] Numero parametri addestrabili per ogni strategia riportato nelle slide
- [ ] Strategia di fine-tuning descritta e motivata
