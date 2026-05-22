"""
Funzioni helper condivise da tutti gli script della pipeline.
"""

import pickle
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ── I/O artefatti ────────────────────────────────────────────────────────────

def save_pickle(obj, path: Path) -> None:
    """
    Serializza obj in un file pickle e stampa dimensione e percorso.

    protocol=pickle.HIGHEST_PROTOCOL usa il protocollo binario più recente
    disponibile nella versione Python corrente: produce file più compatti e
    veloci da leggere rispetto ai protocolli legacy (ASCII o versioni vecchie).
    Importante per file grandi come la matrice dei descrittori (~512 MB).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def load_pickle(path: Path):
    """
    Carica e restituisce l'oggetto serializzato nel file pickle.

    Solleva FileNotFoundError con un messaggio chiaro se il file non esiste:
    più informativo del generico FileNotFoundError di open(), che non indica
    quale artefatto manca nella pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefatto non trovato: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Caricamento immagini ──────────────────────────────────────────────────────

# Estensioni supportate da OpenCV e presenti nei dataset usati:
#   .jpg/.jpeg : AID (immagini aeree JPEG)
#   .tif/.tiff : UC Merced (GeoTIFF a 3 canali RGB, 256×256 px)
#   .png/.bmp  : formati aggiuntivi per test e inferenza su immagini generiche
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def load_image_gray(path: Path) -> np.ndarray | None:
    """
    Carica un'immagine e la converte in scala di grigi (richiesto da SIFT/ORB).

    cv2.IMREAD_GRAYSCALE: carica direttamente in single-channel uint8,
    equivalente a IMREAD_COLOR seguito da cvtColor(BGR2GRAY) ma più efficiente.
    Restituisce None se OpenCV non riesce ad aprire il file (path con caratteri
    Unicode su Windows, file troncato, formato non supportato, ecc.).

    NOTA: str(path) è necessario perché cv2.imread non accetta oggetti Path
    su alcune versioni di OpenCV (< 4.5 circa).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img  # None se il file non è leggibile


def collect_image_paths(root: Path, extensions: set = SUPPORTED_EXTS) -> list[Path]:
    """
    Raccoglie ricorsivamente tutti i percorsi immagine sotto `root`.

    Usata in 01_extract_descriptors.py per raccogliere tutte le immagini AID
    indipendentemente dalla struttura delle sottocartelle.
    Il risultato è ordinato per garantire un ordine deterministico tra esecuzioni.
    """
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in extensions]
    return sorted(paths)


def collect_labeled_paths(root: Path, extensions: set = SUPPORTED_EXTS) -> tuple[list[Path], list[str]]:
    """
    Raccoglie percorsi e label per dataset strutturati come root/<classe>/<img>.

    Assume che ogni sottocartella diretta di root sia una classe.
    I file nella radice stessa vengono ignorati (not class_dir.is_dir()).

    L'ordinamento a due livelli (sorted sulle class_dir, sorted sulle img)
    garantisce che paths e labels abbiano sempre lo stesso ordine tra
    esecuzioni diverse: fondamentale perché X[i] e y[i] devono corrispondere.

    Ritorna
    -------
    paths  : lista di Path alle immagini
    labels : lista di str con il nome della classe per ogni immagine
             (es. "agricultural", "airplane", ...)
    """
    root = Path(root)
    paths, labels = [], []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                paths.append(img_path)
                labels.append(class_dir.name)
    return paths, labels


# ── Visualizzazione ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    """
    Genera e salva (opzionalmente) una heatmap della matrice di confusione.

    Parametri
    ---------
    cm          : np.ndarray (n_classes, n_classes) int — confusion_matrix di sklearn
    class_names : lista di str con i nomi delle classi (ordine = righe/colonne di cm)
    title       : titolo del grafico
    save_path   : se fornito, salva il PNG a questo percorso (dpi=150)

    Note sui parametri grafici:
      figsize=(14, 12) : sufficiente per leggere le 21 etichette senza sovrapposizioni
      fmt="d"          : valori interi nella cella (conteggi, non percentuali)
      cmap="Blues"     : scala di blu: celle vuote (0) bianche, celle piene scure
      linewidths=0.3   : linea sottile tra celle per separare visivamente i quadranti
      rotation=45      : etichette x inclinate per evitare sovrapposizioni con 21 classi
      dpi=150          : risoluzione sufficiente per presentazioni e stampe A4
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    plt.show()


# ── Timer di comodo ───────────────────────────────────────────────────────────

class Timer:
    """
    Context manager che misura il tempo trascorso di un blocco `with`.

    Uso:
        with Timer():
            operazione_lenta()
        # stampa automaticamente "[time] 42.3s"

    time.perf_counter() è il clock ad alta risoluzione di Python:
    più preciso di time.time() (che può avere risoluzione di ~15 ms su Windows)
    e non è influenzato da aggiustamenti dell'orologio di sistema (NTP ecc.).
    """
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        print(f"[time] {self.elapsed:.1f}s")
