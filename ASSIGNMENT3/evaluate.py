# Post-training

# Valuta il modello selezionato (best_model.keras) sul test set di UC Merced,
# senza rieseguire il training. Stampa il classification report per classe
# (precision, recall, f1-score) e le metriche macro-average.

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import config
from dataset import get_uc_datasets

model = tf.keras.models.load_model(config.BEST_MODEL_PATH)

_, _, X_test, y_test, _ = get_uc_datasets()

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(y_true, y_pred,
      target_names=config.UC_CLASS_NAMES, digits=4))