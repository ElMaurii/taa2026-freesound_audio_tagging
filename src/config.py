# -*- coding: utf-8 -*-
"""Parámetros globales del proyecto (audio, espectrograma, modelo, entrenamiento).
Centraliza la configuración para que pipeline.py, models.py y los notebooks
importen siempre los mismos valores.  Basado en la línea base del paper
(DCASE2019 Task 2 / Freesound Audio Tagging 2019)."""
import os
import random
from pathlib import Path

# ============================================================
# Reproducibilidad
# ============================================================
SEED = 42

def set_global_seeds(seed: int = SEED) -> None:
    """Fija la semilla en Python, NumPy y TensorFlow (llamar al inicio)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

# ==========================================
# Rutas (relativas a la raíz del repo; en Colab: /content/<repo>)
# ==========================================
BASE_DIR        = Path(__file__).resolve().parent.parent   # raíz del repo
DATA_RAW        = BASE_DIR / "data" / "raw" # CSVs y audio original (.wav)
DATA_PROCESSED  = BASE_DIR / "data" / "processed"    # espectrogramas precomputados, folds
SAVED_MODELS    = BASE_DIR / "saved_models" # pesos entrenados (gitignored)
EXPERIMENTS     = BASE_DIR / "experiments" # resultados por run

# Archivos de metadatos
CURATED_CSV = DATA_RAW / "train_curated.csv"
NOISY_CSV   = DATA_RAW / "train_noisy.csv"

# ============================================================
# Audio
# ============================================================
SAMPLE_RATE = 44100          # Hz (FSDKaggleono)
N_CHANNELS  = 1              # mono

# ============================================================
# Espectrograma log-mel (STFT 25 ms / salto z)
# ============================================================
STFT_WINDOW_SECONDS = 0.025  # ventana de 25 ms baseline del paper
STFT_HOP_SECONDS = 0.010  # salto de 10 ms baseline del paper
WIN_LENGTH = int(round(STFT_WINDOW_SECONDS * SAMPLE_RATE))  # = 0.025 * 44100 ≈ 1102 muestras
HOP_LENGTH = int(round(STFT_HOP_SECONDS    * SAMPLE_RATE))    # = 0.010 * 44100 ≈ 441  muestras
N_FFT = 2048 # potencia de 2 ≥ WIN_LENGTH

N_MELS  = 96                 # mel bins, si da el tiempo se puede probar con 128 o 64
F_MIN   = 20.0               # Hz
F_MAX   = 20000.0            # Hz (≈ Nyquist)
LOG_OFFSET = 1e-3            # offset antes del log para evitar log(0)

# Ventaneo por audio: cada AUDIO es 1 instancia, troceada en ventanas.
# Batch -> (batch, n_windows[padding+máscara], N_MELS, N_FRAMES, N_CHANNELS)
MAX_AUDIO_SECONDS  = 5.0                                     # se acota la duración por audio
WINDOW_SECONDS     = 1.0                                     # duración de cada ventana
WINDOW_HOP_SECONDS = 0.5                                     # salto entre ventanas
N_FRAMES    = int(round(WINDOW_SECONDS / STFT_HOP_SECONDS))  # = 100 frames por ventana
MAX_WINDOWS = 1 + int(round((MAX_AUDIO_SECONDS - WINDOW_SECONDS) / WINDOW_HOP_SECONDS))  # tope de ventanas
WINDOW_SHAPE = (N_MELS, N_FRAMES, N_CHANNELS)               # (96, 100, 1) por ventana
# n_windows es DINÁMICO por batch (padding + máscara); ordenar los audios por
# duración antes de armar batches reduce el padding desperdiciado.

# ============================================================
# Modelo
# ============================================================
NUM_CLASSES     = 80         # clases 
DROPOUT_RATE    = 0.5        # probar 0.2
LABEL_SMOOTHING = 0.1        # suavizado de etiquetas, probar 0.0 (sin suavizado)
POOLING = "avg"      # "avg" o "max" (pooling final / agregación)

# MobileNet preentrenada (transfer learning con backbone de ImageNet)
PRETRAINED = True             # usar pesos preentrenados
PRETRAINED_TRAINABLE = True   # descongelar backbone (fine-tuning) vs congelar
PRETRAINED_CHANNELS = 3       # ImageNet espera RGB -> repetir el canal mono 3 veces

# ============================================================
# Entrenamiento
# ============================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3 # Adam, probar 1e-4
WEIGHT_DECAY  = 0.0  # decisión del equipo: 0 (opcional 1e-5; con LR bajo casi no aporta)
EPOCHS = None # limitar si demora mucho, por ejemplo 50
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-3 # mejora mínima para contar como "mejoró"
RESTORE_BEST_WEIGHTS = True     # al parar, volver a la mejor época

# Validación: 15% fijo estratificado, SIN cross-validation (ahorra ~4-5x tiempo)
VAL_SPLIT = 0.15

# OJO: LR_DECAY_RATE es REDUNDANTE con ReduceLROnPlateau (ver build_callbacks).
# Usar UN solo mecanismo. Recomendado: LR constante + ReduceLROnPlateau y NO este schedule.
LR_DECAY_RATE  = 0.96

USE_MIXED_PRECISION = True   # float16 en GPU para acelerar

# ============================================================
# Control de entrenamiento (callbacks) — MISMO valor para todo el equipo
# ============================================================
MONITOR_METRIC = "val_lwlrap"   # métrica a vigilar (la de selección)
MONITOR_MODE   = "max"          # lωlrap: más alto = mejor

# Reduce LR on plateau (patience MENOR que el de early stopping)
REDUCE_LR_FACTOR   = 0.5        # multiplica el LR al estancarse
REDUCE_LR_PATIENCE = 3          # < EARLY_STOPPING_PATIENCE -> baja el LR antes de cortar
REDUCE_LR_MIN_LR   = 1e-6       # piso del learning rate


def build_callbacks(checkpoint_path):
    """Callbacks estándar del equipo, construidos desde config. Usar SIEMPRE
    esto en lugar de escribir los valores a mano, así todos comparten el mismo
    patience, factor, etc.

    OJO: el LwlrapCallback (que calcula val_lwlrap) debe ir ANTES que estos en
    la lista del fit(), para que la métrica ya esté en logs cuando se lean."""
    import tensorflow as tf
    patience = EARLY_STOPPING_PATIENCE if EARLY_STOPPING_PATIENCE else 6
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=MONITOR_METRIC, mode=MONITOR_MODE,
            patience=patience, min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=RESTORE_BEST_WEIGHTS),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=MONITOR_METRIC, mode=MONITOR_MODE,
            factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor=MONITOR_METRIC, mode=MONITOR_MODE,
            save_best_only=True),
    ]

# ============================================================
# Augmentación  (obligatoria por consigna; se compara contra "sin aug")
# ============================================================
AUGMENTATION = "specaugment"   # "none" | "specaugment" | "specaugment_mixup"
MIXUP_ALPHA  = 0.2             # parámetro de la Beta para mixup
SPECAUG_TIME_MASKS = 2        # nº de máscaras temporales (SpecAugment)
SPECAUG_FREQ_MASKS = 2        # nº de máscaras de frecuencia (SpecAugment)

# ============================================================
# Búsqueda de hiperparámetros (variar de a un eje — ESCALONADA, no grid)
# Comparación de arquitecturas: usar los valores por defecto de arriba (FIJOS).
# Ablaciones de la MobileNet: variar estos, un eje a la vez, logueando a Comet.
# ============================================================
SEARCH_GRID = {
    "learning_rate":        [1e-3, 1e-4],
    "dropout_rate":         [0.2, 0.5],
    "label_smoothing":      [0.0, 0.1],
    "pretrained_trainable": [True, False],          # experimento estrella de MobileNet
    "augmentation":         ["none", "specaugment", "specaugment_mixup"],  # obligatorio
}