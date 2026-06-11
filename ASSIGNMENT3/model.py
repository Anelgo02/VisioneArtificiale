# Implementazione del blocco residuale CUSTOM
# e dell'architettura completa.

import tensorflow as tf
from tensorflow.keras import layers, Model
import config


# ── Blocco Residuale ──────────────────────────────────────────────────────────
# y = F(x) + x   
# F(x): due conv 3x3 con BN e ReLU
# x:    skip connection (identità se i canali coincidono, conv 1x1 altrimenti)
#
# BN prima di ReLU perché è la scelta originale di He et al. (2016)

def residual_block(x, filters, stride=1):
    """
    Blocco residuale con:
      - Conv 3x3 -> BN -> ReLU
      - Conv 3x3 -> BN
      - skip connection (con proiezione 1x1 se necessario)
      - ReLU finale dopo la somma

    Parametri
    ----------
    x       : tensore di input
    filters : numero di filtri per entrambe le conv
    stride  : stride della prima conv (>1 riduce risoluzione spaziale)
    """
    shortcut = x

    # --- Ramo principale ---
    x = layers.Conv2D(filters, kernel_size=config.KERNEL_SIZE, strides=stride,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=config.KERNEL_SIZE, strides=1,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # --- Skip connection ---
    # Se stride != 1 oppure il numero di canali cambia, non possiamo sommare
    # direttamente: le dimensioni non coincidono.

    # Soluzione: conv 1x1 con lo stesso stride per proiettare lo shortcut.
    # (uso_bias=False perché la BN che segue ha il proprio bias implicito)
    shortcut_channels = shortcut.shape[-1]
    if stride != 1 or shortcut_channels != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
                                 padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # --- Somma e attivazione finale ---
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


# ── Architettura completa ─────────────────────────────────────────────────────
# Struttura:
#   Input
#   -> Conv iniziale (stem) per portare l'immagine a una rappresentazione base
#   -> Stage 1: blocco residuale (FILTERS_STAGE_1 canali)
#   -> MaxPool  (dimezza risoluzione spaziale)
#   -> Stage 2: blocco residuale (FILTERS_STAGE_2 canali)
#   -> MaxPool
#   -> Stage 3: blocco residuale opzionale (FILTERS_STAGE_3 canali)
#   -> MaxPool
#   -> Flatten  (NON global average pooling come richiesto)
#   -> MLP classificatore (FC + Dropout + FC + ... + output softmax)
#


def build_model(num_classes, input_shape=None):
    """
    Costruisce la CNN residuale custom.

    Parametri
    ----------
    num_classes : int - numero di classi in output (30 per AID, 21 per UC)
    input_shape : tuple - (H, W, C), default da config
    """
    if input_shape is None:
        input_shape = (*config.IMG_SIZE, config.CHANNELS)

    inputs = tf.keras.Input(shape=input_shape)

    # --- Stem: primo layer convoluzionale ---
    # Serve a estrarre feature di base (bordi, texture)
    # prima di entrare nei blocchi residuali. 
    x = layers.Conv2D(config.FILTERS_STAGE_1, kernel_size=config.KERNEL_SIZE,
                      strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # --- Stage 1: primo blocco residuale ---
    x = residual_block(x, filters=config.FILTERS_STAGE_1)
    x = layers.MaxPooling2D(pool_size=2)(x)   # 128->64 (o 64->32 ecc.)

    # --- Stage 2: secondo blocco residuale ---
    # Raddoppiamo i filtri: meno risoluzione spaziale, più canali semantici.
    # È la convenzione standard nelle ResNet (collo di bottiglia nella dimensione spaziale
    # compensato da più feature channels).
    x = residual_block(x, filters=config.FILTERS_STAGE_2)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # --- Stage 3: terzo blocco residuale ---
    x = residual_block(x, filters=config.FILTERS_STAGE_3)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # --- Flatten ---
    # Trasforma (H', W', C) in un vettore 1D.
    x = layers.Flatten()(x)

    # --- MLP ---
    for units in config.MLP_UNITS:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(config.DROPOUT_RATE)(x)

    # Output: softmax su num_classes neuroni.
    # Produce una distribuzione di probabilità su tutte le classi,
    # necessaria con categorical cross-entropy.
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="ResidualCNN_custom")
    return model


# ── Utility: sostituisce la testa per il fine-tuning ─────────────────────────
def replace_head(pretrained_model, num_classes_new, finetune_strategy):
    """
    Prende un modello pre-addestrato su AID (30 classi) e sostituisce
    il layer di output con uno compatibile con UC Merced (21 classi).

    Parametri
    ----------
    pretrained_model  : keras.Model addestrato su AID
    num_classes_new   : int - numero di nuove classi (21)
    finetune_strategy : str - "frozen", "partial", "full"

    Ritorna il modello modificato pronto per il fine-tuning.
    """
    # Prendi tutto il backbone tranne l'ultimo layer (softmax su 30 classi)
    backbone_output = pretrained_model.layers[-2].output

    # Nuova testa per 21 classi
    new_output = layers.Dense(num_classes_new, activation='softmax',
                               name='output_uc')(backbone_output)

    new_model = Model(inputs=pretrained_model.input, outputs=new_output,
                      name="ResidualCNN_finetuned")

    # Applica la strategia di congelamento
    if finetune_strategy == "frozen":
        # Congela TUTTO il backbone, addestra solo l'ultimo Dense
        for layer in new_model.layers[:-1]:
            layer.trainable = False

    elif finetune_strategy == "partial":
        # Congela i primi 2/3 della rete, lascia liberi gli ultimi layer conv + MLP
        total = len(new_model.layers)
        freeze_until = int(total * 0.6)   # congela il 60% iniziale
        for layer in new_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in new_model.layers[freeze_until:]:
            layer.trainable = True

    elif finetune_strategy == "full":
        # Tutto addestrabile, ma useremo un LR molto basso
        for layer in new_model.layers:
            layer.trainable = True

    # Stampa un riepilogo dei parametri addestrabili
    trainable_params = sum(tf.size(w).numpy() for w in new_model.trainable_weights)
    total_params     = sum(tf.size(w).numpy() for w in new_model.weights)
    print(f"\nStrategia: {finetune_strategy}")
    print(f"Parametri totali:       {total_params:,}")
    print(f"Parametri addestrabili: {trainable_params:,}")
    print(f"Parametri congelati:    {total_params - trainable_params:,}\n")

    return new_model
