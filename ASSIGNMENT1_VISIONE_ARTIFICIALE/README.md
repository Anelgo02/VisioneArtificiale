# Progetto: Registrazione Immagini Multimodali via Mutua Informazione

Questo progetto implementa un sistema di allineamento automatico per coppie di immagini provenienti da sensori diversi (RGB e Termico). Il sistema utilizza la massimizzazione della **Mutua Informazione (MI)** per trovare la rototraslazione ottimale.

---

## 🛠️ Requisiti e Installazione

Il progetto richiede Python 3.8 o superiore. Per installare le librerie necessarie è possibile eseguire:

```bash
pip install -r requirements.txt
```

**Le dipendenze principali sono:**
* **numpy**: Gestione matriciale e calcolo istogrammi.
* **opencv-python**: Caricamento immagini e trasformazioni geometriche.
* **scipy**: Algoritmi di ottimizzazione non lineare.

---

## 🚀 Come utilizzare il progetto

### Allineamento di una coppia singola (`main.py`)
è possibile utilizzare questo script per allineare due immagini.

```bash
python main.py --ref "percorso/immagine_fissa.png" --mov "percorso/immagine_da_allineare.png"
```

**Risultati attesi:**
* Parametri di trasformazione ($T_x, T_y, \theta$) stampati a terminale.
* `risultato_allineato.png`: L'immagine allineata (con bordi neri tecnici).
* `mappa_differenza.png`: Differenza assoluta per verificare la sovrapposizione dei bordi.

### Riproduzione Esperimenti (`validazione.py`)
è possibile utilizzare questo script per processare l'intero Validation Set e confrontare le performance (MAE) al variare dei bin e dei metodi.

```bash
python validazione.py
```

### Validazione Test Set (`valutazione_test.py`)
Questo script esegue l’ottimizzazione con la configurazione migliore trovata (Powell, 64 bins), valuta il Test Set e calcola il MAE finale su traslazione e rotazione. Salva immagini allineate e mappe differenza in `RISULTATI_TEST`.

```bash
python valutazione_test.py
```

### esportazione immagini di validazione (`val_images.py`)
Questo script processa il Validation Set e genera output visivi (aligned e diff) in `RISULTATI_VAL`.

```bash
python val_images.py
```

---

## 📂 Struttura del Codice
* **[main.py](main.py)**: Interfaccia a riga di comando per l'allineamento di singole coppie.
* **[metrics.py](metrics.py)**: Funzioni core per il calcolo di Entropia e Mutua Informazione.
* **[registrator.py](registrator.py)**: Gestione delle trasformazioni spaziali e della funzione obiettivo.
* **[validazione.py](validazione.py)**: Script per la validazione massiva sul set di validazione (MAE, confronto bin/metodi).
* **[valutazione_test.py](valutazione_test.py)**: Script per la stessa configurazione ottimale sul test set (stima errori finali e salvataggio risultati).
* **[val_images.py](val_images.py)**: Script di supporto per generare immagini allineate e mappe diff in output per tutte le coppie.

---

## 🧠 Metodologia e Scelte Tecniche
In seguito a una fase di testing sul Validation Set, è stata isolata la configurazione che garantisce il minor Errore Medio Assoluto (MAE):

1. **Ottimizzatore (Powell)**: Scelto per la sua robustezza nel gestire rotazioni senza richiedere il calcolo del gradiente (metodo derivative-free).
2. **Istogramma (64 bins)**: Identificato come il valore ottimale per evitare l'instabilità statistica dovuta al rumore dei sensori termici.
3. **Pre-processing (Gaussian Blur 7x7)**: Applicato per regolarizzare la superficie della Mutua Informazione, ampliando il raggio di cattura dell'algoritmo e riducendo i minimi locali.

**Formula della Mutua Informazione:**
$$MI(A, B) = H(A) + H(B) - H(A, B)$$

---

## 📝 Note sui Risultati
* **Bordi Neri**: Sono il risultato geometrico della rototraslazione. Indicano le aree in cui l'immagine originale non era presente dopo lo spostamento.
* **Mappa Differenza**: Trattandosi di immagini multimodali, la mappa evidenzia differenze cromatiche dovute alla diversa natura dei sensori. L'allineamento è considerato corretto quando i contorni delle strutture principali coincidono.

