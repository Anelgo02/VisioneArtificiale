# Note per le slide — Assignment 2 BoW
# Fabio Gulotta — Visione Artificiale

Questo file raccoglie, script per script, le scelte progettuali e le funzioni
rilevanti da citare nelle slide. Va aggiornato man mano che completiamo la pipeline.

---

## SLIDE 1 — Titolo e obiettivo

**Titolo suggerito**: Classificazione di immagini aeree con Bag-of-Visual-Words

**Obiettivo in una frase**:
Classificare immagini aeree RGB del dataset UC Merced (21 classi) usando
l'approccio classico BoW: descrittori locali SIFT → vocabolario visuale →
istogrammi → classificatori shallow (SVM-RBF, Random Forest).

**Cosa dire**:
- Due dataset usati con ruoli diversi: AID per costruire il vocabolario (etichette ignorate),
  UC Merced per addestrare e valutare il classificatore.
- Approccio pre-deep learning, interpretabile e con costi computazionali contenuti.

---

## SLIDE 2 — Pipeline BoW (schema a blocchi)

```
AID (10.000 img)
      │
      ▼
  SIFT su ogni img          ← 01_extract_descriptors.py
  max 100 desc/img
      │
      ▼
  ~1.000.000 desc (128-d)
      │
      ▼
  MiniBatchKMeans           ← 02_build_vocabulary.py
  K ∈ {50, 100, 500}
      │
      ▼
  Vocabolario (K centroidi)
      │
UC Merced (2.100 img)       ← 03_compute_bow.py
      │
      ▼
  Istogramma BoW (K bin)
  normalizzato L2
      │
      ▼
  SVM-RBF / Random Forest   ← 04_cross_validation.py
  3-fold StratifiedKFold
      │
      ▼
  Modello finale            ← 05_train_final.py
```

---

## SLIDE 3 — Dataset

### AID (Aerial Image Dataset)
- Usato per: costruire il vocabolario visuale (etichette ignorate)
- 30 classi, 10.000 immagini JPG
- Distribuzione non uniforme tra le classi (normale per AID)
- Immagini RGB 600×600 px

### UC Merced Land Use Dataset
- Usato per: addestramento e valutazione del classificatore
- 21 classi, 2.100 immagini TIFF, esattamente 100 per classe (dataset bilanciato)
- Classi: airport, beach, buildings, chaparral, forest, freeway, golf course...
- Immagini RGB 256×256 px

**Scelta chiave**: vocabolario da AID (dataset più grande e vario) → visual words
più generali e rappresentative rispetto a costruirlo su UC Merced stesso.

---

## SLIDE 4 — Fase 1: estrazione SIFT e costruzione vocabolario

### Script: `01_extract_descriptors.py`

**Cosa fa**:
Itera su tutte le 10.000 immagini AID, estrae descrittori SIFT da ognuna,
campiona al massimo 100 per immagine, accumula tutto in una matrice unica.

**Funzioni principali**:
- `cv2.SIFT_create()` — crea il rilevatore SIFT di OpenCV
- `sift.detectAndCompute(img, None)` — trova keypoint e calcola descrittori (128-d float32)
- `rng.choice(len(descs), size=100, replace=False)` — campionamento casuale riproducibile
- `np.vstack(all_descriptors)` — unisce tutto in matrice (N × 128)

**Scelte e motivazioni**:
| Scelta | Motivazione |
|--------|-------------|
| SIFT invece di ORB | SIFT produce float a 128-d: compatibile con K-Means euclideo. ORB è binario: la media di vettori binari non è definita → K-Means non si applica naturalmente |
| Max 100 desc/img | 10.000 img × 100 desc × 128 float32 × 4 byte ≈ 512 MB → gestibile in RAM |
| Grayscale | SIFT analizza struttura locale (gradienti): il colore non aggiunge informazione rilevante |
| seed=42 nel campionamento | Riproducibilità: stesso campione su ogni macchina e ogni run |

**Output**: `models/descriptors_aid.pkl` — np.ndarray shape (≈1.000.000, 128)

---

### Script: `02_build_vocabulary.py`

**Cosa fa**:
Carica la matrice di descrittori e addestra MiniBatchKMeans per ognuno
dei tre valori di K. I K centroidi risultanti sono le "visual words".

**Funzioni principali**:
- `MiniBatchKMeans(n_clusters=K, batch_size=10_000, random_state=42)`
- `kmeans.fit(descriptors)` — clustering su ~1M descrittori
- `kmeans.inertia_` — somma delle distanze quadratiche: misura qualità clustering

**Scelte e motivazioni**:
| Scelta | Motivazione |
|--------|-------------|
| MiniBatchKMeans invece di KMeans | KMeans standard su 1M righe deve calcolare distanze N×K in memoria → lento e memory-intensive. MiniBatch processa a lotti da 10.000: stessa qualità, tempi drasticamente ridotti |
| K ∈ {50, 100, 500} | Richiesto dalla consegna. K piccolo → parole generiche; K grande → parole specifiche ma istogrammi sparsi |
| Oggetto KMeans salvato (non solo centroidi) | In fase 2 si usa `kmeans.predict()` direttamente: più comodo e garantisce stessa metrica di distanza |

**Output**: `models/vocabulary_K50.pkl`, `vocabulary_K100.pkl`, `vocabulary_K500.pkl`

---

## SLIDE 5 — Fase 2: rappresentazione BoW

### Script: `03_compute_bow.py`

**Cosa fa**:
Per ogni valore di K, carica il vocabolario addestrato e itera su tutte le
2.100 immagini UC Merced. Per ognuna:
1. Estrae descrittori SIFT
2. Assegna ogni descrittore alla visual word più vicina (`kmeans.predict`)
3. Conta le occorrenze con `np.bincount` → istogramma di K bin
4. Normalizza L2 l'intera matrice in un'unica chiamata sklearn

**Funzioni principali**:
- `kmeans.predict(descs)` — assegna ogni descrittore al centroide più vicino,
  restituisce array di indici es. `[3, 42, 3, 17, 91, ...]`
- `np.bincount(words, minlength=k)` — conta le occorrenze di ogni indice → istogramma
- `normalize(X, norm="l2")` — normalizza L2 tutta la matrice in una sola chiamata
- Fallback: immagini senza keypoint ricevono un istogramma di zeri (non crashano)

**Scelte e motivazioni**:
| Scelta | Motivazione |
|--------|-------------|
| Normalizzazione L2 | Senza normalizzazione, immagini con molti keypoint dominano per magnitudine indipendentemente dalla classe. L2 è standard in letteratura BoW con SVM-RBF |
| L2 invece di L1 | L1 interpreta l'istogramma come distribuzione di probabilità. L2 è preferita con kernel RBF e SVM |
| Normalizzazione sulla matrice intera | Più efficiente che normalizzare riga per riga nel loop. Stessa semantica |
| Salvataggio di `class_names` nel pickle | Evita di ricostruire la lista delle classi nei prossimi script |

**Verifica automatica**: dopo la normalizzazione lo script stampa `np.linalg.norm(X[0])`.
Deve essere `~1.0000`. Se non lo è, c'è un bug nella normalizzazione.

**Output** (per ogni K): `models/bow_ucmerced_K{k}.pkl`
→ dizionario con `X` (2100×K, float32), `y` (2100,), `class_names` (lista 21 classi)

---

## SLIDE 5b — Fase 4: training finale e inferenza

### Script: `05_train_final.py`

**Cosa fa**:
Legge `cv_results.csv`, identifica la combinazione vincente (descriptor, K, classificatore),
poi esegue tre operazioni:
1. Genera la matrice di confusione su uno split 70/30 stratificato (per visualizzazione)
2. Riaddestra il modello vincente sull'intero dataset UC Merced (2100 immagini)
3. Salva il modello finale e una copia del vocabolario vincente con nome fisso

**Funzioni principali**:
- `df.loc[df["f1_mean"].idxmax()]` — seleziona automaticamente la riga migliore da cv_results.csv
- `StratifiedShuffleSplit(test_size=0.30)` — split 70/30 stratificato per la CM
- `CLASSIFIER_FACTORY[clf_name]()` — dizionario di lambda che crea istanze fresche del classificatore
- `shutil.copy(voc_src, FINAL_VOCAB_FILE)` — copia il vocabolario vincente come `vocabulary_final.pkl`

**Scelte e motivazioni**:
| Scelta | Motivazione |
|--------|-------------|
| Due training separati (CM + finale) | Il modello per la CM viene valutato su dati mai visti (30% held-out). Il modello finale viene addestrato su tutto il dataset perché va in produzione: più dati → performance migliori sul campo |
| `CLASSIFIER_FACTORY` con lambda | scikit-learn non permette di riusare un oggetto già addestrato. Le lambda creano due istanze fresche per i due training |
| `vocabulary_final.pkl` con nome fisso | `inference.py` carica sempre lo stesso file senza dover sapere quale K ha vinto |
| Selezione automatica da CSV | Nessuna scelta manuale: il modello è determinato interamente dai risultati della CV |

**Output**:
- `models/classifier_final.pkl` — classificatore addestrato su tutto UC Merced
- `models/vocabulary_final.pkl` — copia del vocabolario vincente
- `results/confusion_matrix.png` — heatmap 21×21 classi

---

### Script: `inference.py`

**Cosa fa**:
Script da riga di comando che classifica una singola immagine arbitraria.
Ricostruisce la stessa pipeline usata in training:
1. Carica vocabolario e classificatore finale
2. Legge da `cv_results.csv` quale tipo di descrittore ha vinto
3. Estrae i descrittori, costruisce l'istogramma BoW L2-normalizzato
4. Predice la classe

**Uso**:
```bash
python inference.py path/alla/immagine.jpg
# Classe predetta   : forest
```

**Scelte e motivazioni**:
| Scelta | Motivazione |
|--------|-------------|
| `get_descriptor_type()` legge da CSV | Garantisce che inference usi lo stesso descrittore del modello salvato, senza hardcoding |
| `kmeans.n_clusters` per ricavare K | Non serve salvare K separatamente: è già nell'oggetto KMeans |
| Fallback a vettore zero se no keypoint | L'immagine viene comunque classificata invece di crashare |
| Top-3 classi con probabilità | Mostra le alternative più probabili (solo se il classificatore espone `predict_proba`) |

---

## SLIDE 6 — Fase 3: classificatori e protocollo CV

### Script: `04_cross_validation.py` ← DA COMPLETARE

**Classificatori scelti**:

| Classificatore | Parametri | Motivazione |
|----------------|-----------|-------------|
| SVM kernel RBF | default sklearn | Standard storico per BoW. Il kernel RBF proietta implicitamente in spazio non lineare ad alta dimensione, ideale per istogrammi normalizzati L2 |
| Random Forest | default sklearn | Paradigma completamente diverso: ensemble di alberi, non basato su distanze o kernel. Il confronto è informativo |

**Perché questi due e non altri**:
SVM-RBF e RF "stressano" la rappresentazione in modi opposti.
SVM guarda la geometria globale; RF usa soglie locali su singole feature.
Scegliere SVM-lineare + SVM-RBF sarebbe stato troppo simile.

**Protocollo di valutazione**:
- `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
- Stratified: ogni fold mantiene la proporzione delle 21 classi
- Metriche: accuracy, precision, recall, F1-score (tutti macro-averaged)
- Macro-average: tutte le 21 classi pesano uguale (corretto con dataset bilanciato)

**Output**: `results/cv_results.csv` con righe (K, classificatore) e colonne (accuracy_mean, accuracy_std, f1_mean, f1_std, ...)

---

## SLIDE 7 — Risultati: tabella comparativa ✅ COMPLETATA

> Dati reali da `results/cv_results.csv` — esecuzione del 15 Maggio 2026

| K | Classificatore | Accuracy (mean ± std) | F1 macro (mean ± std) |
|---|----------------|-----------------------|-----------------------|
| 50 | SVM-RBF | 66.6% ± 1.4% | 66.2% ± 1.5% |
| 50 | Random Forest | 62.5% ± 0.2% | 61.2% ± 0.3% |
| 100 | SVM-RBF | 68.2% ± 1.1% | 67.8% ± 1.1% |
| 100 | Random Forest | 61.8% ± 0.5% | 60.5% ± 0.6% |
| **500** | **SVM-RBF** | **71.6% ± 1.2%** | **71.3% ± 1.0%** |
| 500 | Random Forest | 60.1% ± 0.5% | 58.2% ± 0.7% |

**Modello vincente**: SIFT + K=500 + SVM-RBF

**Cosa commentare**:
- SVM-RBF migliora monotonicamente con K: più visual words → rappresentazione più ricca → separazione migliore nello spazio kernel RBF
- Random Forest degrada con K crescente: istogrammi K=500 sono molto sparsi (la maggior parte dei bin è zero); gli alberi faticano a trovare split utili su feature sparse ad alta dimensione
- Std basse (≤ 1.5%) → risultati stabili tra i fold, non fluke
- Il risultato è coerente con la letteratura BoW: SVM-RBF con vocabolario grande è la combinazione classicamente vincente

---

## SLIDE 8 — Matrice di confusione ✅ COMPLETATA

> PNG in `results/confusion_matrix.png` — SIFT K=500 SVM-RBF, split 70/30 stratificato

**Cosa commentare**:
- Classi con texture molto caratteristica (beach, forest, chaparral) → diagonale alta, pochi errori
- Confusioni attese e spiegabili:
  - buildings ↔ denseresidential: stessa struttura visiva (tetti ravvicinati), BoW non ha informazione di scala
  - mediumresidential ↔ sparseresidential: differenza solo nella densità, istogrammi BoW simili
  - freeway ↔ storageanks: entrambe con strutture geometriche ripetitive (corsie/serbatoi)
- Connessione con BoW: le classi confuse hanno visual words simili perché le texture locali si assomigliano;
  il BoW non "vede" la struttura globale della scena, solo l'aggregato di patch locali
- La CM è su dati held-out (30% mai visti in training): è una valutazione onesta

---

## SLIDE 9 — Problemi riscontrati e soluzioni

> Sezione da aggiornare durante lo sviluppo quando si incontrano problemi reali.

| Problema | Soluzione adottata |
|----------|--------------------|
| Warning Pylance su cv2/tqdm in VS Code | Falso positivo: pacchetti installati nel conda env. Risolto selezionando interprete corretto |
| *(aggiungere durante sviluppo)* | |

---

## SLIDE 10 — Possibili miglioramenti

**Limitazioni del BoW classico**:
- Nessuna informazione spaziale: due immagini con le stesse visual words in posizioni
  diverse producono lo stesso istogramma
- Hard assignment: ogni descrittore viene assegnato a UNA sola visual word (anche se
  è vicino al confine tra due centroidi)
- Vocabolario fisso: non si adatta al task di classificazione

**Miglioramenti possibili**:
| Tecnica | Cosa migliora |
|---------|---------------|
| Spatial Pyramid Matching | Aggiunge informazione sulla posizione spaziale delle visual words |
| Soft assignment | Ogni descrittore contribuisce a più visual words con peso proporzionale alla distanza |
| TF-IDF weighting | Pesa le visual words rare più di quelle comuni (analogo al TF-IDF nel text mining) |
| Fisher Vectors / VLAD | Codifiche più ricche che catturano statistiche di primo e secondo ordine |
| CNN features (transfer learning) | Feature estratte da reti pre-addestrate: molto più discriminative, ma black-box |

**Quando preferire BoW a CNN**:
Dataset molto piccoli, necessità di interpretabilità, risorse computazionali limitate,
ambienti embedded senza GPU.

---

## SLIDE +1 — Riferimenti e strumenti utilizzati

**Dataset**:
- AID: Xia et al., "AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification", IEEE TGRS 2017
- UC Merced: Yang & Newsam, "Bag-of-visual-words and spatial extensions for land-use classification", ACM GIS 2010

**Librerie**:
- OpenCV 4.10 — estrazione SIFT
- scikit-learn 1.4 — MiniBatchKMeans, SVM, Random Forest, StratifiedKFold
- NumPy 1.26, matplotlib, seaborn, tqdm

**Strumenti AI**:
- Claude (Anthropic) — discussione progettuale, revisione scelte architetturali,
  scrittura e debugging degli script Python, comprensione teorica degli algoritmi.
  Ogni scelta progettuale è stata discussa e validata dallo studente.

---

*Ultimo aggiornamento: 15 Maggio 2026 — pipeline eseguita, slide 7 e 8 completate con dati reali*
