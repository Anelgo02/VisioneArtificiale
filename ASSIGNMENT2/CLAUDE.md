# CLAUDE.md — Assignment 2: Classificazione BoW

Documento di stato per Claude Code. Aggiornare ad ogni sessione.

## Progetto

Classificatore per immagini aeree RGB su 21 classi (UC Merced) tramite **Bag-of-Visual-Words**:
vocabolario costruito con SIFT + MiniBatchKMeans su AID, classificatori SVM-RBF e Random Forest,
valutazione con 3-fold StratifiedKFold. Dettaglio completo in `PROJECT_GUIDELINES.md`.

- **Studente**: Fabio Gulotta
- **Scadenza**: 27 Maggio 2026 ore 23:59
- **Env conda**: `visioneArtificiale`

---

## Stato attuale — 19 Maggio 2026

### Completato
- [x] `requirements.txt` — dipendenze pin-nate
- [x] `config.py` — parametri centralizzati. Funzioni file includono `desc_type` per supportare SIFT e ORB
- [x] `utils.py` — helper: save/load pickle, caricamento immagini, collect_labeled_paths, plot_confusion_matrix, Timer
- [x] `00_check_datasets.py` — verifica integrità dataset (conteggio classi, test OpenCV, test SIFT)
- [x] `01_extract_descriptors.py` — SIFT (e opzionalmente ORB) su AID, max 100 desc/img
- [x] `02_build_vocabulary.py` — MiniBatchKMeans per ogni (desc_type, K), K ∈ {50, 100, 500}
- [x] `03_compute_bow.py` — istogrammi BoW L2-normalizzati per UC Merced, per ogni (desc_type, K)
- [x] `04_cross_validation.py` — 3 assi di confronto: descrittore × K × classificatore
- [x] `05_train_final.py` — training finale sul modello vincente, CM su split 70/30, salva artefatti + classification_report
- [x] `inference.py` — script singola immagine → classe predetta
- [x] `SLIDE_NOTES.md` — note dettagliate per la presentazione (tutte le slide pronte con dati reali)
- [x] **Pipeline completa ESEGUITA** (15 Maggio sul fisso Ryzen 5800X3D)
- [x] `results/cv_results.csv` — risultati 3-fold CV per tutte le combinazioni (K × classificatore)
- [x] `results/confusion_matrix.png` — heatmap 21×21, modello vincente, split 70/30
- [x] `results/classification_report.txt` — precision/recall/F1 per singola classe (aggiunto 19 Maggio)

### Risultati chiave
- **Modello vincente**: SIFT + K=500 + SVM-RBF
- **CV**: Accuracy 71.6% ± 1.2%, F1 macro 71.3% ± 1.0%
- **Split 30% held-out**: Accuracy 73.0%, F1 macro 73%
- **Classi migliori**: chaparral (0.98), parkinglot (0.98), harbor (0.95), agricultural (0.93)
- **Classi peggiori**: sparseresidential (0.51), storagetanks (0.51), baseballdiamond (0.56)

### Da fare
- [ ] **Presentazione PDF** — struttura pronta in `SLIDE_NOTES.md`, entro 25–26 Maggio
  - Slide 7: tabella CV + metriche aggregate del modello vincente (dati in `cv_results.csv`)
  - Slide 8: matrice di confusione (`results/confusion_matrix.png`)
  - Slide 9: problemi riscontrati (encoding cp1252 su Windows risolto con reconfigure UTF-8)

### Note ORB
- Il codice supporta ORB (conversione uint8→float32 in `01_extract_descriptors.py`) ma NON è stato eseguito
- `config.py` ha `DESCRIPTOR_TYPES = ["SIFT"]` — ORB è disabilitato
- La conversione float32 è un'approssimazione (Euclidea su byte, non Hamming su bit): da dichiarare se si include
- Decisione: lasciare solo SIFT, citare ORB come possibile miglioramento in slide 10

---

## Struttura file

```
ASSIGNMENT2/
├── CLAUDE.md                  ← questo file
├── PROJECT_GUIDELINES.md      ← decisioni progettuali e motivazioni
├── SLIDE_NOTES.md             ← note per le slide (aggiornare dopo esecuzione)
├── config.py                  ← parametri centralizzati
├── utils.py                   ← funzioni helper condivise
├── requirements.txt
├── .gitignore
│
├── 00_check_datasets.py       ← [ESEGUITO] verifica dataset
├── 01_extract_descriptors.py  ← SIFT/ORB su AID → descriptors_aid_{type}.pkl
├── 02_build_vocabulary.py     ← MiniBatchKMeans → vocabulary_{type}_K{k}.pkl
├── 03_compute_bow.py          ← istogrammi BoW → bow_ucmerced_{type}_K{k}.pkl
├── 04_cross_validation.py     ← 3-fold CV → cv_results.csv
├── 05_train_final.py          ← training finale → classifier_final.pkl + CM
├── inference.py               ← python inference.py <path_img>
│
├── Classificazione di Immagini attraverso BoW.docx  ← bozza presentazione
│
├── datasets/                  ← NON in git — inserire manualmente sul fisso
│   ├── AID/<classe>/*.jpg
│   └── UCMerced/Images/<classe>/*.tif
├── models/                    ← NON in git — generato dalla pipeline
└── results/                   ← NON in git — generato dalla pipeline
```

---

## Assi di confronto sperimentale

| Asse | Opzioni | Obbligatorio |
|------|---------|--------------|
| Tipo descrittore | SIFT, ORB | ORB opzionale — abilitare in `config.py` |
| Dimensione vocabolario K | 50, 100, 500 | Sì |
| Classificatore | SVM-RBF, Random Forest | Sì |

Per abilitare ORB: in `config.py` cambiare `DESCRIPTOR_TYPES = ["SIFT"]` in `["SIFT", "ORB"]`.
Tutti gli script si adattano automaticamente senza altre modifiche.

---

## Decisioni chiave (sintesi)

| Scelta | Alternativa scartata | Motivo |
|--------|----------------------|--------|
| SIFT (float 128-d) | ORB (binario) | K-Means euclideo non definito su vettori binari |
| MiniBatchKMeans | KMeans pieno | ~1M descrittori: troppo lento/memory con KMeans standard |
| Max 100 desc/img AID | Tutti i desc | Bilancio RAM (≈512 MB) / diversità vocabolario |
| Norm L2 istogrammi | L1 / nessuna | Standard per SVM-RBF; evita bias da numero di keypoint |
| SVM-RBF + RandomForest | SVM-lin + SVM-RBF | Confronto informativo tra paradigmi diversi |
| StratifiedKFold(3) | KFold semplice | Garantisce proporzione classi per fold (richiesto consegna) |
| CM su split 70/30 | CM su training | La CM deve mostrare errori su dati mai visti |
| RANDOM_SEED = 42 | — | Riproducibilità totale tra macchine diverse |

---

## Comandi rapidi

```bash
conda activate visioneArtificiale
cd ".../ASSIGNMENT2"

python 00_check_datasets.py       # verifica dataset (già eseguito)
python 01_extract_descriptors.py  # ~15-20 min sul fisso
python 02_build_vocabulary.py     # ~5-15 min per K
python 03_compute_bow.py          # ~2-5 min per K
python 04_cross_validation.py     # ~10-30 min per combinazione
python 05_train_final.py          # ~5 min
python inference.py <path_img>    # test su singola immagine
```

---

## Note operative

- Gli script numerati sono **idempotenti**: se l'artefatto `.pkl` esiste già, saltano il ricalcolo.
- Per forzare il ricalcolo: cancellare il `.pkl` corrispondente in `models/`.
- Se si cambia macchina: copiare solo `models/*.pkl` e `results/` — i dataset vanno reinseriti.
- Dopo esecuzione: aggiornare `SLIDE_NOTES.md` con i valori reali di slide 7, 8, 10.
