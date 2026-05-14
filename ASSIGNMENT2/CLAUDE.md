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

## Stato attuale — 13 Maggio 2026

### Completato
- [x] `requirements.txt` — dipendenze pin-nate
- [x] `config.py` — tutti i parametri centralizzati (percorsi, K values, seed, path artefatti)
- [x] `utils.py` — helper: save/load pickle, caricamento immagini, collect_labeled_paths, plot_confusion_matrix, Timer
- [x] `00_check_datasets.py` — verifica integrità dataset (conteggio classi, test OpenCV, test SIFT)
- [x] `.gitignore` — esclude `datasets/`, `models/`, `results/`
- [x] **Fase 0 ESEGUITA** — dataset verificati e pronti:
  - AID: 30 classi, 10.000 immagini (distribuzione non uniforme, normale per AID)
  - UC Merced: 21 classi, 2.100 immagini (100 per classe, perfettamente bilanciato)
  - OpenCV e SIFT funzionanti su entrambi i dataset

### Da fare (in ordine)
- [ ] **Fase 1a** (`01_extract_descriptors.py`): SIFT su tutte le immagini AID, max 100 desc/img, salva `models/descriptors_aid.pkl`
- [ ] **Fase 1b** (`02_build_vocabulary.py`): MiniBatchKMeans per K ∈ {50, 100, 500}, salva `models/vocabulary_K{k}.pkl`
- [ ] **Fase 2** (`03_compute_bow.py`): istogrammi BoW L2-normalizzati per UC Merced, salva `models/bow_ucmerced_K{k}.pkl`
- [ ] **Fase 3** (`04_cross_validation.py`): 3-fold CV, SVM-RBF + Random Forest, salva `results/cv_results.csv`
- [ ] **Fase 4** (`05_train_final.py`): training finale sul modello vincente, salva `models/classifier_final.pkl`
- [ ] **Inferenza** (`inference.py`): script singola immagine → classe predetta
- [ ] **Slide** (`presentazione.pdf`): 10+1 slide, struttura in `PROJECT_GUIDELINES.md` §12

---

## Struttura file

```
ASSIGNMENT2/
├── CLAUDE.md                  ← questo file
├── PROJECT_GUIDELINES.md      ← decisioni progettuali e motivazioni
├── config.py                  ← parametri centralizzati (modificare qui percorsi e K)
├── utils.py                   ← funzioni helper condivise
├── requirements.txt
├── .gitignore
│
├── 00_check_datasets.py       ← [PROSSIMO] verifica dataset prima di partire
├── 01_extract_descriptors.py  ← [DA CREARE]
├── 02_build_vocabulary.py     ← [DA CREARE]
├── 03_compute_bow.py          ← [DA CREARE]
├── 04_cross_validation.py     ← [DA CREARE]
├── 05_train_final.py          ← [DA CREARE]
├── inference.py               ← [DA CREARE]
│
├── datasets/                  ← NON in git — inserire manualmente
│   ├── AID/<classe>/*.jpg
│   └── UCMerced/Images/<classe>/*.tif
├── models/                    ← NON in git — generato dalla pipeline
└── results/                   ← NON in git — generato dalla pipeline
```

---

## Decisioni chiave (sintesi)

| Scelta | Alternativa scartata | Motivo |
|--------|----------------------|--------|
| SIFT (float 128-d) | ORB (binario) | K-Means con distanza euclidea non funziona su binari |
| MiniBatchKMeans | KMeans pieno | ~1M descrittori: troppo lento/memory con KMeans standard |
| Max 100 desc/img AID | Tutti i desc | Bilancio RAM (≈512 MB) / diversità vocabolario |
| Norm L2 istogrammi | L1 / nessuna | Standard per SVM-RBF; evita bias da numero di keypoint |
| SVM-RBF + RandomForest | SVM-lin + SVM-RBF | Confronto informativo tra paradigmi diversi |
| StratifiedKFold(3) | KFold semplice | Garantisce proporzione classi per fold (richiesto consegna) |
| RANDOM_SEED = 42 | — | Riproducibilità totale tra macchine diverse |

---

## Comandi rapidi

```bash
conda activate visioneArtificiale
cd "D:\Documenti\Visione Artificiale\ASSIGNMENT2"

python 00_check_datasets.py       # verifica dataset
python 01_extract_descriptors.py  # ~10-20 min sul fisso
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
