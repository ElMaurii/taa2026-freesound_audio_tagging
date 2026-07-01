# -*- coding: utf-8 -*-
"""Métrica oficial lωlrap (label-weighted label-ranking average precision).

Implementación de referencia de Dan Ellis (dpwe@google.com), del Colab oficial
de DCASE2019 Task 2:
    https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8

Se mantuvo la implementación original (solo se modernizaron los dtype, porque
np.int/np.bool/np.float fueron eliminados de NumPy >= 1.24) y se agregaron:
  - calculate_overall_lwlrap(): el escalar directo (lo que se reporta).
  - aggregate_masked(): agrega ventanas -> clip (la métrica es a nivel de clip).
  - LwlrapCallback: para usar lωlrap como métrica de selección en Keras.

lωlrap es la métrica del leaderboard: usarla para seleccionar modelos, NO
accuracy ni F1."""
import numpy as np
import sklearn.metrics


# ============================================================
# Implementación de referencia (NumPy) — exacta
# ============================================================
def _one_sample_positive_class_precisions(scores, truth):
  """Calculate precisions for each true class for a single sample.

  Args:
    scores: np.array of (num_classes,) giving the individual classifier scores.
    truth: np.array of (num_classes,) bools indicating which classes are true.

  Returns:
    pos_class_indices: np.array of indices of the true classes for this sample.
    pos_class_precisions: np.array of precisions corresponding to each of those
      classes.
  """
  num_classes = scores.shape[0]
  pos_class_indices = np.flatnonzero(truth > 0)
  # Only calculate precisions if there are some true classes.
  if not len(pos_class_indices):
    return pos_class_indices, np.zeros(0)
  # Retrieval list of classes for this sample.
  retrieved_classes = np.argsort(scores)[::-1]
  # class_rankings[top_scoring_class_index] == 0 etc.
  class_rankings = np.zeros(num_classes, dtype=int)
  class_rankings[retrieved_classes] = range(num_classes)
  # Which of these is a true label?
  retrieved_class_true = np.zeros(num_classes, dtype=bool)
  retrieved_class_true[class_rankings[pos_class_indices]] = True
  # Num hits for every truncated retrieval list.
  retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
  # Precision of retrieval list truncated at each hit, in order of pos_labels.
  precision_at_hits = (
      retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
      (1 + class_rankings[pos_class_indices].astype(float)))
  return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
  """Calculate label-weighted label-ranking average precision.

  Arguments:
    truth: np.array of (num_samples, num_classes) giving boolean ground-truth
      of presence of that class in that sample.
    scores: np.array of (num_samples, num_classes) giving the classifier-under-
      test's real-valued score for each class for each sample.

  Returns:
    per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
      class.
    weight_per_class: np.array of (num_classes,) giving the prior of each
      class within the truth labels.  Then the overall unbalanced lwlrap is
      simply np.sum(per_class_lwlrap * weight_per_class)
  """
  assert truth.shape == scores.shape
  num_samples, num_classes = scores.shape
  # Space to store a distinct precision value for each class on each sample.
  # Only the classes that are true for each sample will be filled in.
  precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
  for sample_num in range(num_samples):
    pos_class_indices, precision_at_hits = (
      _one_sample_positive_class_precisions(scores[sample_num, :],
                                            truth[sample_num, :]))
    precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
        precision_at_hits)
  labels_per_class = np.sum(truth > 0, axis=0)
  weight_per_class = labels_per_class / float(np.sum(labels_per_class))
  # Form average of each column, i.e. all the precisions assigned to labels in
  # a particular class.
  per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                      np.maximum(1, labels_per_class))
  return per_class_lwlrap, weight_per_class


def calculate_overall_lwlrap_sklearn(truth, scores):
  """Calculate the overall lwlrap using sklearn.metrics.lrap."""
  # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
  sample_weight = np.sum(truth > 0, axis=1)
  nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
  overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0,
      scores[nonzero_weight_sample_indices, :],
      sample_weight=sample_weight[nonzero_weight_sample_indices])
  return overall_lwlrap


# ============================================================
# Atajos / utilidades del proyecto
# ============================================================
def calculate_overall_lwlrap(truth, scores):
  """lωlrap global (un único número). Es lo que se reporta en el informe."""
  truth = np.asarray(truth)
  scores = np.asarray(scores)
  per_class, weight = calculate_per_class_lwlrap(truth, scores)
  return float(np.sum(per_class * weight))


def aggregate_masked(window_scores, mask, mode="mean"):
  """Agrega las predicciones de las ventanas de cada clip -> score por clip.

  La métrica es a nivel de CLIP, pero la red predice por ventana. Si en
  inferencia obtenés un score por ventana, usá esto para agregarlas
  respetando el padding (la agregación ignora las ventanas de relleno).

  Args:
    window_scores: (n_clips, n_windows, num_classes) scores por ventana.
    mask: (n_clips, n_windows) con 1 = ventana real, 0 = padding.
    mode: "mean" (promedio) o "max" (máximo).

  Returns:
    (n_clips, num_classes) score agregado por clip.
  """
  window_scores = np.asarray(window_scores, dtype=float)
  m = np.asarray(mask, dtype=float)[..., None]      # (n_clips, n_windows, 1)
  if mode == "max":
    masked = np.where(m > 0, window_scores, -np.inf)
    return masked.max(axis=1)
  # mean (ignorando padding)
  summed = (window_scores * m).sum(axis=1)
  counts = np.maximum(m.sum(axis=1), 1.0)
  return summed / counts


# ============================================================
# Callback de Keras — lωlrap sobre validación al final de cada época
# (se define solo si TensorFlow está disponible)
# ============================================================
try:
  import tensorflow as tf

  class LwlrapCallback(tf.keras.callbacks.Callback):
    """Calcula lωlrap sobre el conjunto de validación al terminar cada época
    y lo registra en logs['val_lwlrap'], para monitorearlo con EarlyStopping /
    ModelCheckpoint (mode='max').

    IMPORTANTE: este callback debe ir ANTES que EarlyStopping/ModelCheckpoint
    en la lista del fit(), para que 'val_lwlrap' ya esté en logs cuando esos lo
    lean. Asume que el modelo entrega un score por CLIP (la agregación de
    ventanas ocurre dentro del modelo, vía pooling enmascarado).

    Args:
      val_data: tf.data.Dataset que entrega (x, y), o tupla NumPy (X, y).
    """
    def __init__(self, val_data):
      super().__init__()
      self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
      logs = logs if logs is not None else {}
      y_pred = self.model.predict(self.val_data, verbose=0)
      if isinstance(self.val_data, tuple):
        y_true = self.val_data[1]
      else:
        y_true = np.concatenate([y.numpy() for _, y in self.val_data], axis=0)
      lwlrap = calculate_overall_lwlrap(y_true, y_pred)
      logs['val_lwlrap'] = lwlrap
      print(f" — val_lwlrap: {lwlrap:.4f}")

except ImportError:                # entorno sin TensorFlow: solo NumPy/sklearn
  tf = None
  LwlrapCallback = None


# ============================================================
# Test de validación (solo al ejecutar el archivo directamente)
# ============================================================
if __name__ == "__main__":
  rng = np.random.RandomState(42)          # semilla fija -> reproducible
  num_samples, num_labels = 100, 20

  truth = rng.rand(num_samples, num_labels) > 0.5
  truth[0:1, :] = False                     # algún clip sin etiquetas
  scores = rng.rand(num_samples, num_labels)

  per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(truth, scores)
  nativo = float(np.sum(per_class_lwlrap * weight_per_class))
  sk = calculate_overall_lwlrap_sklearn(truth, scores)
  print("lwlrap (per-class) =", nativo)
  print("lwlrap (sklearn)   =", sk)
  print("calculate_overall  =", calculate_overall_lwlrap(truth, scores))

  # El port es correcto si nativo y sklearn coinciden
  assert abs(nativo - sk) < 1e-6, "El port de lwlrap NO coincide con sklearn"
  print("OK: el port de lwlrap coincide con sklearn.")
