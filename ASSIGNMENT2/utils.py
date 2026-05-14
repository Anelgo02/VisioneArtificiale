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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def load_pickle(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefatto non trovato: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Caricamento immagini ──────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def load_image_gray(path: Path) -> np.ndarray | None:
    """Carica un'immagine e la converte in scala di grigi (per SIFT)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img  # None se il file non è leggibile


def collect_image_paths(root: Path, extensions: set = SUPPORTED_EXTS) -> list[Path]:
    """Raccoglie tutti i percorsi immagine sotto `root` ricorsivamente."""
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in extensions]
    return sorted(paths)


def collect_labeled_paths(root: Path, extensions: set = SUPPORTED_EXTS) -> tuple[list[Path], list[str]]:
    """
    Raccoglie percorsi e label per dataset strutturati come root/<classe>/<img>.
    Restituisce (paths, labels).
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
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        print(f"[time] {self.elapsed:.1f}s")
