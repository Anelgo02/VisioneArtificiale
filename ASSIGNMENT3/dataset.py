
# Gestisce: caricamento immagini con OpenCV, split stratificato, augmentation.

# Usiamo tf.data per l'efficienza del pipeline durante il training.

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import config


# ── Caricamento paths e labels ────────────────────────────────────────────────

def load_paths_and_labels(dataset_dir):
    """
    Scansiona dataset_dir aspettandosi struttura:
        dataset_dir/
            classe_1/
                img1.jpg
                img2.jpg
            classe_2/
                ...

    Ritorna:
        paths  : lista di path assoluti alle immagini
        labels : lista di interi (indice di classe)
        classes: lista ordinata dei nomi di classe
    """
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    paths, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

    print(f"Dataset: {dataset_dir}")
    print(f"  Classi trovate: {len(classes)}")
    print(f"  Immagini totali: {len(paths)}")
    return paths, labels, classes


# ── Split stratificato ────────────────────────────────────────────────────────
# "Stratificato" significa che ogni classe mantiene la stessa proporzione in
# train/val/test. Critico con dataset sbilanciati o con poche immagini per classe.

def split_dataset(paths, labels, seed=config.RANDOM_SEED):
    """
    Split 70/15/15 stratificato.
    Viene fatto UNA SOLA VOLTA e i path vengono salvati (o usati fissi con seed).
    """
    # Prima split: separa test (15%) dal resto (85%)
    paths_trainval, paths_test, labels_trainval, labels_test = train_test_split(
        paths, labels,
        test_size=config.TEST_RATIO,
        stratify=labels,
        random_state=seed
    )

    # Seconda split: separa val (15% del totale = ~17.6% del trainval) dal train
    val_size_relative = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_trainval, labels_trainval,
        test_size=val_size_relative,
        stratify=labels_trainval,
        random_state=seed
    )

    print(f"  Train:      {len(paths_train)} immagini")
    print(f"  Validation: {len(paths_val)} immagini")
    print(f"  Test:       {len(paths_test)} immagini")
    return (paths_train, labels_train,
            paths_val,   labels_val,
            paths_test,  labels_test)


# ── Caricamento e preprocessing con OpenCV ────────────────────────────────────
# OpenCV usato per manipolazioni

def load_and_preprocess_image(path, target_size=config.IMG_SIZE):
    """
    Carica un'immagine con OpenCV, la ridimensiona e normalizza in [0, 1].
    Nota: OpenCV carica in BGR; convertiamo in RGB per coerenza con Keras.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))   # cv2: (W, H)
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_batch(paths, labels, num_classes):
    """Carica un intero split in memoria come array numpy."""
    X = np.array([load_and_preprocess_image(p) for p in paths])
    y = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    return X, y


# ── Augmentation (solo su training set) ──────────────────────────────────────
# Perché solo sul training set? Applicarla a val/test causerebbe data leakage:
# valuteremmo il modello su dati artificialmente modificati, non su dati reali.

def augment_image(image, label):
    """
    Augmentation leggera:
      - flip orizzontale casuale
      - rotazione casuale ±15°
      - variazione di luminosità
    Queste trasformazioni non cambiano la classe semantica (un aeroporto
    ruotato è ancora un aeroporto).
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ── Costruzione tf.data.Dataset ───────────────────────────────────────────────
# tf.data è più efficiente di caricare tutto in memoria per dataset grandi (AID).

def make_tf_dataset(X, y, batch_size, augment=False, shuffle=False):
    """
    Crea un tf.data.Dataset a partire da array numpy X, y.

    Parametri
    ----------
    X        : array (N, H, W, C) float32
    y        : array (N, num_classes) one-hot
    augment  : bool - se True applica augmentation (solo training set)
    shuffle  : bool - se True mescola ogni epoca
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Funzione di alto livello per l'AID (pre-training) ────────────────────────

def get_aid_datasets():
    """
    Prepara train/val per AID usando split 70/30 (non ha val set nativo).
    """
    paths, labels, classes = load_paths_and_labels(config.AID_DIR)
    num_classes = len(classes)

    # Split 70/30 stratificato
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths, labels,
        test_size=0.30,
        stratify=labels,
        random_state=config.RANDOM_SEED
    )
    print(f"  AID Train: {len(paths_train)} | Val: {len(paths_val)}")

    X_train, y_train = preprocess_batch(paths_train, labels_train, num_classes)
    X_val,   y_val   = preprocess_batch(paths_val,   labels_val,   num_classes)

    ds_train = make_tf_dataset(X_train, y_train,
                                batch_size=config.PRETRAIN_BATCH,
                                augment=True, shuffle=True)
    ds_val   = make_tf_dataset(X_val,   y_val,
                                batch_size=config.PRETRAIN_BATCH)
    return ds_train, ds_val, num_classes


# ── Funzione di alto livello per UC Merced ────────────────────────────────────

def get_uc_datasets():
    """
    Prepara train/val/test per UC Merced con split FISSO 70/15/15.
    Ritorna anche X_test, y_test come array per la valutazione finale.
    """
    paths, labels, classes = load_paths_and_labels(config.UC_DIR)
    num_classes = len(classes)

    (paths_train, labels_train,
     paths_val,   labels_val,
     paths_test,  labels_test) = split_dataset(paths, labels)

    X_train, y_train = preprocess_batch(paths_train, labels_train, num_classes)
    X_val,   y_val   = preprocess_batch(paths_val,   labels_val,   num_classes)
    X_test,  y_test  = preprocess_batch(paths_test,  labels_test,  num_classes)

    ds_train = make_tf_dataset(X_train, y_train,
                                batch_size=config.FINETUNE_BATCH,
                                augment=True, shuffle=True)
    ds_val   = make_tf_dataset(X_val,   y_val,
                                batch_size=config.FINETUNE_BATCH)

    return ds_train, ds_val, X_test, y_test, num_classes
