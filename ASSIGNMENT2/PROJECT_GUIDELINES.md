# PROJECT GUIDELINES — Assignment 2: Classificazione BoW

> Documento di riferimento operativo per lo sviluppo dell'assignment.
> Tutte le scelte progettuali sono raccolte qui con la relativa giustificazione.
> Da rileggere prima di ogni sessione di lavoro e prima dell'orale.

**Studente**: Fabio Gulotta
**Corso**: Visione Artificiale
**Scadenza**: 27 Maggio 2026, ore 23:59
**Inizio sviluppo**: 13 Maggio 2026

---

## 1. Obiettivo del progetto

Sviluppare un classificatore per immagini aeree RGB su 21 classi del dataset UC Merced, utilizzando l'approccio classico **Bag-of-Visual-Words (BoW)**:

1. costruzione di un vocabolario di K parole visuali tramite clustering di descrittori locali estratti dal dataset **AID** (etichette ignorate),
2. rappresentazione di ogni immagine UC Merced come istogramma di occorrenze delle visual words,
3. addestramento di almeno 2 classificatori shallow,
4. valutazione tramite 3-fold cross-validation stratificata,
5. selezione del miglior modello (combinazione di K e classificatore).

---

## 2. Filosofia di sviluppo

Tre principi che guidano ogni scelta:

1. **Riproducibilità**. Ogni esperimento deve essere ri-eseguibile esattamente con gli stessi risultati. Random seed fissati ovunque.
2. **Persistenza degli artefatti intermedi**. Ogni fase salva il suo output su disco. Mai ricalcolare ciò che è già stato calcolato.
3. **Tracciabilità delle scelte**. Ogni decisione non banale è documentata, qui o nel codice, con la motivazione.

---

## 3. Stack tecnologico

| Componente | Versione | Motivazione della scelta |
|------------|----------|--------------------------|
| Python | 3.11 | Stabile, ben supportato da tutte le librerie. 3.12 ancora ha qualche incompatibilità. |
| NumPy | 1.26.x | NumPy 2.0 ha breaking changes, evitiamoli. Installato via conda per avere MKL. |
| OpenCV | 4.10.x (`opencv-python`) | SIFT disponibile dal modulo principale dalla 4.4+. NO `opencv-contrib-python`. |
| scikit-learn | 1.4.x | Standard per ML classico. Fornisce MiniBatchKMeans, SVM, RandomForest, StratifiedKFold, metriche. |
| matplotlib + seaborn | 3.8 / 0.13 | Visualizzazioni per matrici di confusione e grafici comparativi. |
| tqdm | 4.66 | Barre di progresso. Indispensabile per loop lunghi. |
| joblib | 1.4 | Parallelizzazione su CPU multi-core (estrazione SIFT). |

**Ambiente**: conda env `cv_bow` dedicato. NumPy/sklearn installati via conda (per le BLAS ottimizzate), OpenCV via pip dentro l'env conda.

---

## 4. Hardware e strategia di esecuzione

| Macchina | Specifiche | Ruolo |
|----------|------------|-------|
| Fisso | Ryzen 5800X3D, 16 GB RAM | Esecuzione pipeline pesante: estrazione descrittori, K-Means, cross-validation. |
| MacBook Air M2 | M2, 8 GB RAM | Sviluppo codice, debugging su sottoinsiemi piccoli, scrittura slide. |

**Trasferimento artefatti**: i pickle (vocabolario, classificatore) sono piccoli e portabili tra le due macchine. I dataset stanno solo sul fisso.

---

## 5. Struttura del progetto

```
assignment_2_bow/
│
├── PROJECT_GUIDELINES.md      # questo file
├── config.py                  # parametri centralizzati
├── utils.py                   # funzioni helper riusate
├── requirements.txt           # dipendenze
├── README.md                  # come eseguire
│
├── 01_extract_descriptors.py  # Fase 1a: SIFT su AID
├── 02_build_vocabulary.py     # Fase 1b: K-Means
├── 03_compute_bow.py          # Fase 2: istogrammi BoW per UC Merced
├── 04_cross_validation.py     # Fase 3: 3-fold CV + selezione modello
├── 05_train_final.py          # Fase 4: training finale + salvataggio
├── inference.py               # script di inferenza (richiesto da consegna)
│
├── models/                    # artefatti .pkl (in .gitignore se uso git)
└── results/                   # log, csv, matrici confusione
```

**Convenzioni**:

- **Script numerati** = ordine di esecuzione esplicito.
- **Un solo `utils.py`** per le funzioni helper (load_pickle, save_pickle, plot_confusion_matrix, ecc.).
- **`config.py` unico** per tutti i parametri (paths, K values, random_seed, ecc.).
- **Dataset FUORI dalla cartella progetto** (in `~/datasets/AID` e `~/datasets/UCMerced` o equivalente).

---

## 6. Decisioni progettuali principali

### 6.1 Descrittori locali: SIFT (non ORB)

**Scelta**: SIFT.

**Motivazione**:
- SIFT produce descrittori float a 128 dimensioni, K-Means standard funziona bene con la distanza euclidea.
- ORB produce descrittori binari (32 byte): la media aritmetica di vettori binari non è binaria, quindi K-Means standard non è la scelta naturale (servirebbe k-medoids o varianti con Hamming distance, complicazione non necessaria).
- SIFT è invariante a scala, rotazione e moderate variazioni di illuminazione: caratteristiche desiderabili per immagini aeree con orientamenti arbitrari.

**ORB come opzionale**: solo se avanza tempo per il punto opzionale della consegna.

### 6.2 Limitazione dei descrittori

**Scelta**: massimo 100 descrittori per immagine di AID, campionati casualmente se SIFT ne trova di più.

**Motivazione**:
- 10.000 immagini × 100 descrittori = 1.000.000 descrittori totali.
- 1.000.000 × 128 float32 × 4 byte ≈ 512 MB di RAM. Gestibile sul fisso da 16 GB.
- La consegna stessa suggerisce 100-200 descrittori/immagine o ≤100k totali. Sceglierei l'approccio "molte immagini × pochi descrittori" perché favorisce la diversità del vocabolario rispetto alla profondità per singola immagine.

### 6.3 Clustering: MiniBatchKMeans (non KMeans)

**Scelta**: `MiniBatchKMeans` con `batch_size=10000`.

**Motivazione**:
- KMeans standard su matrici da ~1M righe è lento e memory-intensive (deve mantenere distanze N×K in memoria).
- MiniBatchKMeans processa i descrittori in batch: drasticamente più veloce, qualità del clustering molto vicina a quella di KMeans pieno.
- Stesso paradigma usato in praticamente tutte le pipeline BoW production-grade.

### 6.4 Valori di K da testare

**Scelta**: K ∈ {50, 100, 500} (come richiesto dalla consegna).

**Aspettative qualitative**:
- K=50: vocabolario troppo piccolo, probabile sotto-segmentazione (parole visuali troppo generiche).
- K=100: punto di partenza ragionevole.
- K=500: vocabolario ricco, ma rischio di sovra-frammentazione e sparsità degli istogrammi su 21 classi.

### 6.5 Normalizzazione degli istogrammi

**Scelta**: **L2** come normalizzazione di default.

**Motivazione**:
- Senza normalizzazione, immagini con molti keypoint dominerebbero la magnitudine dell'istogramma indipendentemente dalla classe → il classificatore confonderebbe "quantità di struttura" con "tipo di scena".
- L2 è preferita rispetto a L1 quando si usa SVM con kernel RBF o lineare, perché molti kernel sono pensati per vettori L2-normalizzati. Tipica in letteratura BoW.
- L1 (somma=1) ha senso quando l'istogramma è interpretato come distribuzione di probabilità; per la classificazione tramite SVM L2 è la scelta standard.

### 6.6 Classificatori scelti

**Scelta**: **SVM con kernel RBF** + **Random Forest**.

**Motivazione del confronto**:
- SVM-RBF: standard storico per BoW, kernel implicitamente proietta in spazio non-lineare, molto performante con istogrammi normalizzati.
- Random Forest: paradigma completamente diverso (ensemble di alberi, non parametrico, non basato su kernel). Il confronto è informativo perché stressano proprietà diverse della rappresentazione BoW.
- Scelta "informata" rispetto a SVM-lineare + SVM-RBF (troppo simili) o k-NN + Logistic Regression (entrambi lineari/distance-based).

**Iperparametri**: default di scikit-learn (suggerito dalla consegna stessa per semplicità). Eventuale tuning solo se accuracy bassa.

### 6.7 Cross-validation: StratifiedKFold

**Scelta**: `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`.

**Motivazione**:
- La consegna richiede esplicitamente bilanciamento delle classi nei fold. StratifiedKFold lo garantisce: ogni fold mantiene la proporzione delle classi del dataset originale.
- UC Merced ha 100 immagini per classe (bilanciato), ma StratifiedKFold è comunque buona pratica perché:
  1. previene problemi se il dataset risultasse leggermente sbilanciato,
  2. il professore vedrà che hai scelto il metodo corretto, non un KFold semplice.
- 3 fold = ~700 immagini in test, ~1400 in training. Adeguato per il numero di classi (21).

### 6.8 Random seed

**Scelta**: `RANDOM_SEED = 42` globale, definito in `config.py`, usato in:
- campionamento dei descrittori,
- inizializzazione di MiniBatchKMeans (`random_state=42`),
- StratifiedKFold (`random_state=42`),
- inizializzazione SVM e RandomForest (`random_state=42`).

**Motivazione**: senza seed fissati, ogni run produce risultati leggermente diversi. Per riproducibilità (e per non avere figure di matrice confusione diverse nelle slide rispetto al codice consegnato), seed bloccato.

---

## 7. Metriche di valutazione

Per ogni combinazione (K, classificatore) calcoliamo, mediate sui 3 fold:

- **Accuracy** (globale).
- **Precision, Recall, F1-score**: macro-average (media non pesata sulle 21 classi). Macro perché tutte le classi pesano uguale, coerente con dataset bilanciato.
- **Matrice di confusione**: solo per il modello finale (combinazione vincente), salvata come PNG per le slide.

**Output di Fase 3**: tabella CSV con righe (K, classificatore) e colonne (accuracy_mean, accuracy_std, f1_mean, f1_std, ...).

---

## 8. Cronoprogramma

| Fase | Date | Output |
|------|------|--------|
| Fase 0 — Setup ambiente + download dataset | 13–14 Maggio | conda env funzionante, dataset scaricati |
| Fase 1 — Estrazione descrittori + vocabolario | 15–17 Maggio | `descriptors_aid.pkl`, `vocabulary_K{50,100,500}.pkl` |
| Fase 2 — Rappresentazione BoW UC Merced | 18–19 Maggio | Per ogni K: matrice X (2100×K) + vettore y |
| Fase 3 — Cross-validation | 20–22 Maggio | `cv_results.csv`, tabella comparativa |
| Fase 4 — Training finale + inferenza | 23–24 Maggio | `classifier_final.pkl`, `inference.py` testato |
| Fase 5 — Slide PDF | 25–26 Maggio | `presentazione.pdf` (max 10 + 1 slide) |
| Consegna | 27 Maggio entro 23:59 | Codice + PDF |

**Buffer**: 2 giorni di margine pianificato all'interno delle date. Se una fase salta, sappiamo di averli.

---

## 9. Rischi noti e mitigazioni

| Rischio | Probabilità | Mitigazione |
|---------|-------------|-------------|
| Immagine UC Merced senza keypoint SIFT | Bassa | Try/except in fase di estrazione, istogramma di zeri fallback |
| Cluster K-Means vuoto | Bassa | MiniBatchKMeans ha gestione interna; verifica post-clustering |
| Out-of-memory su clustering K=500 | Bassa con MiniBatch | Riduzione `batch_size` se necessario |
| Tempi di SVM-RBF eccessivi su K=500 | Media | Iperparametri default; se serve, ridurre `C` o usare `LinearSVC` |
| TIFF di UC Merced non letti correttamente da OpenCV | Bassa | Test su 1 immagine in Fase 0 prima di procedere |
| Slide non finite in tempo | Media | Inizio scrittura Fase 5 già il 25 Maggio, buffer di 2 giorni |

---

## 10. Domande tipiche da orale (preparazione)

Domande che il professore potrebbe fare. Le risposte vanno costruite progressivamente durante lo sviluppo.

1. Perché hai scelto SIFT e non ORB?
2. Cos'è una "visual word" e perché ha senso analogizzarla a una parola in un documento?
3. Cosa cambia tra MiniBatchKMeans e KMeans? Quando uno è preferibile all'altro?
4. Perché normalizzi gli istogrammi? Cosa succederebbe se non lo facessi?
5. Perché L2 e non L1? Avresti potuto usare TF-IDF?
6. Perché SVM-RBF + Random Forest, e non SVM-lineare + SVM-RBF?
7. Cosa significa "stratified" in StratifiedKFold? Perché usarlo qui?
8. Cosa ti aspetteresti dalla matrice di confusione? Quali classi confonderà di più?
9. Come scala il sistema all'aumentare di K? Pro e contro di K grande/piccolo.
10. Limiti del BoW classico rispetto a CNN moderne? Quando preferiresti BoW?

---

## 11. Note operative per ogni sessione

Prima di ogni sessione di lavoro:

1. Attivare l'env: `conda activate cv_bow`.
2. Verificare in che fase siamo: leggere l'ultimo log in `results/logs/`.
3. Non rilanciare fasi già completate: gli artefatti in `models/` sono persistenti.
4. Aggiornare questo file se prendiamo una decisione nuova non prevista.

---

## 12. Slide di presentazione (struttura prevista)

10 slide + 1 obbligatoria. Bozza:

1. Titolo + obiettivo del progetto.
2. Pipeline BoW (schema visivo a blocchi).
3. Dataset (AID per vocabolario, UC Merced per classificazione).
4. Fase 1: estrazione SIFT e clustering K-Means.
5. Fase 2: rappresentazione BoW (immagine → istogramma).
6. Fase 3: classificatori scelti e protocollo CV.
7. Risultati: tabella accuracy/F1 al variare di K e classificatore.
8. Matrice di confusione del modello migliore.
9. Problemi riscontrati e soluzioni.
10. Possibili miglioramenti (TF-IDF, soft assignment, spatial pyramid, CNN features).

+1. **Slide obbligatoria**: riferimenti, risorse usate, uso di strumenti AI (Claude per discussione progettuale, scrittura codice di supporto, comprensione teorica).

---

*Ultima revisione: 13 Maggio 2026*
