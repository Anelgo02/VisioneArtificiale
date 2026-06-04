# train.py
# Gestisce le due strategie di addestramento e la model selection finale.
#
# Strategia 1: pre-training su AID -> fine-tuning su UC
# Strategia 2: training da zero su UC
# Model selection: confronto su validation set (F1-score macro)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, ConfusionMatrixDisplay)

import config
from model import build_model, replace_head
from dataset import get_aid_datasets, get_uc_datasets

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Callback comuni ───────────────────────────────────────────────────────────
# EarlyStopping: obbligatorio per assignment.
#   monitor='val_loss': se la loss di validazione non migliora per `patience` epoche,
#   si ferma. restore_best_weights recupera i pesi dell'epoca migliore.
# ModelCheckpoint: salva automaticamente il migliore.

def get_callbacks(model_path, patience):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]


# ── Plot delle curve di training ──────────────────────────────────────────────
# Obbligatorio nelle slide: permette di diagnosticare overfitting visivamente.
# Overfitting tipico: training loss scende, validation loss risale dopo un minimo.

def plot_history(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['loss'],     label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['accuracy'],     label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curva salvata: {save_path}")


# ── STRATEGIA 1a: Pre-training su AID ────────────────────────────────────────

def pretrain_on_aid():
    print("\n" + "="*60)
    print("FASE 1: PRE-TRAINING SU AID")
    print("="*60)

    ds_train, ds_val, num_classes = get_aid_datasets()
    model = build_model(num_classes=num_classes)

    # Categorical cross-entropy come richiesto dal task.
    # Perché? Con one-hot encoding + softmax, questa loss misura la divergenza KL
    # tra la distribuzione predetta e quella vera (one-hot = distribuzione degenere).
    # Adam: ottimizzatore adattivo, buona scelta di default per training da zero.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.PRETRAIN_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=config.PRETRAIN_EPOCHS,
        callbacks=get_callbacks(config.PRETRAINED_MODEL_PATH,
                                config.PRETRAIN_PATIENCE)
    )

    plot_history(history, "Pre-training AID",
                 "results/pretrain_AID_curves.png")

    print(f"Modello pre-addestrato salvato: {config.PRETRAINED_MODEL_PATH}")
    return model


# ── STRATEGIA 1b: Fine-tuning su UC Merced ────────────────────────────────────

def finetune_on_uc(pretrained_model, ds_train, ds_val):
    print("\n" + "="*60)
    print(f"FASE 2: FINE-TUNING SU UC - Strategia: {config.FINETUNE_STRATEGY}")
    print("="*60)

    model = replace_head(pretrained_model,
                         num_classes_new=config.UC_CLASSES,
                         finetune_strategy=config.FINETUNE_STRATEGY)

    # LR più basso per il fine-tuning: fondamentale.
    # Un LR alto riscrive i pesi appresi su AID, vanificando il pre-training.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=config.FINETUNE_EPOCHS,
        callbacks=get_callbacks(config.FINETUNED_MODEL_PATH,
                                config.FINETUNE_PATIENCE)
    )

    plot_history(history, f"Fine-tuning UC ({config.FINETUNE_STRATEGY})",
                 "results/finetune_UC_curves.png")

    print(f"Modello fine-tunato salvato: {config.FINETUNED_MODEL_PATH}")
    return model


# ── STRATEGIA 2: Training da zero su UC ───────────────────────────────────────

def train_from_scratch(ds_train, ds_val):
    print("\n" + "="*60)
    print("STRATEGIA 2: TRAINING DA ZERO SU UC MERCED")
    print("="*60)

    model = build_model(num_classes=config.UC_CLASSES)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.SCRATCH_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=config.SCRATCH_EPOCHS,
        callbacks=get_callbacks(config.SCRATCH_MODEL_PATH,
                                config.SCRATCH_PATIENCE)
    )

    plot_history(history, "Training da zero UC",
                 "results/scratch_UC_curves.png")

    print(f"Modello scratch salvato: {config.SCRATCH_MODEL_PATH}")
    return model


# ── Valutazione su validation set (model selection) ───────────────────────────
# Il confronto avviene sul VALIDATION set, non sul test set.
# Il test set viene toccato UNA SOLA VOLTA alla fine, sul modello selezionato.

def evaluate_on_val(model, ds_val, label):
    """Calcola F1-macro sul validation set per la model selection."""
    y_true, y_pred = [], []
    for X_batch, y_batch in ds_val:
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n[{label}] F1-score macro (val): {f1:.4f}")
    return f1


# ── Valutazione finale sul test set ───────────────────────────────────────────

def evaluate_on_test(model, X_test, y_test):
    """
    Calcola Accuracy, Precision, Recall, F1-macro, Confusion Matrix sul test set.
    Questo viene chiamato UNA SOLA VOLTA sul modello selezionato.
    """
    print("\n" + "="*60)
    print("VALUTAZIONE FINALE SUL TEST SET")
    print("="*60)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test, verbose=1), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=config.UC_CLASS_NAMES,
                                 digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=config.UC_CLASS_NAMES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    plt.title("Confusion Matrix - Test Set (held-out)")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_test.png", dpi=150)
    plt.close()
    print("Confusion matrix salvata: results/confusion_matrix_test.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Carica i dataset UC (split fisso, usato per entrambe le strategie)
    ds_train_uc, ds_val_uc, X_test, y_test, _ = get_uc_datasets()

    # -- Strategia 1 --
    pretrained = pretrain_on_aid()
    model_s1   = finetune_on_uc(pretrained, ds_train_uc, ds_val_uc)

    # -- Strategia 2 --
    model_s2 = train_from_scratch(ds_train_uc, ds_val_uc)

    # -- Model selection su validation set --
    f1_s1 = evaluate_on_val(model_s1, ds_val_uc, "Strategia 1 (fine-tuning)")
    f1_s2 = evaluate_on_val(model_s2, ds_val_uc, "Strategia 2 (scratch)")

    if f1_s1 >= f1_s2:
        best_model = model_s1
        print(f"\nSelezionato: Strategia 1 (F1={f1_s1:.4f} vs {f1_s2:.4f})")
    else:
        best_model = model_s2
        print(f"\nSelezionato: Strategia 2 (F1={f1_s2:.4f} vs {f1_s1:.4f})")

    best_model.save(config.BEST_MODEL_PATH)
    print(f"Miglior modello salvato: {config.BEST_MODEL_PATH}")

    # -- Valutazione finale (una sola volta sul test set) --
    evaluate_on_test(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
