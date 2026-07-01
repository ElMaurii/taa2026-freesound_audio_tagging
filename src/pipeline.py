# Funciones de carga y conversión de audio a espectrograma

# ============================================================
# 1. Etiquetas (vocabulario multi-hot)
# ============================================================

import pandas as pd
import numpy as np
from src import config as C

def cargar_vocabulario(csv_path=None):
    """ Devueleve la lista determinista de las 80 clases.
    La deriva de train_curated.csv y la ordena alfabéticamente 
    para que sea la misma para todos.
    """
    cvs_path = csv_path or C.CURATED_CSV
    df = pd.read_csv(cvs_path)
    clases = set()
    for tags in df["labels"]:
        for c in tags.split(","):
            clases.add(c.strip())
    clases = sorted(list(clases))
    assert len(clases) == C.NUM_CLASSES, f"Se esperaban {C.NUM_CLASSES} clases, pero se encontraron {len(clases)}"
    return clases

def etiquetas_a_multihot(labels_str, clase_a_idx):
    """Convierte una cadena de etiquetas separadas por comas a un vector multihot.

    Args:
        labels_str: cadena de etiquetas, p.ej. "dog,cat,bird"
        clase_a_idx: diccionario que mapea cada clase a su índice en el vector multihot

    Returns:
        np.array de forma (NUM_CLASSES,) con 1s en las posiciones de las clases presentes y 0s en las ausentes.
    """
    multihot = np.zeros(len(clase_a_idx), dtype=np.float32)
    for label in labels_str.split(","):
        multihot[clase_a_idx[label.strip()]] = 1.0
    return multihot

def construir_matriz_etiquetas(df, clases):
    """Construye la matriz de etiquetas multihot para un DataFrame dado.

    Args:
        df: DataFrame con una columna "labels" que contiene las etiquetas separadas por comas.
        clases: lista de todas las clases posibles.

    Returns:
        np.array de forma (len(df), len(clases)) con las etiquetas multihot.
    """
    clase_a_idx = {clase: idx for idx, clase in enumerate(clases)}

    return np.stack([etiquetas_a_multihot(labels, clase_a_idx) for labels in df["labels"]], axis=0)

def multihot_a_etiquetas(y, clases, umbral=0.5):
    """Convierte un vector multihot a una lista de etiquetas.

    Args:
        y: np.array de forma (NUM_CLASSES,) con probabilidades (vector de prediccion).
        clases: lista de todas las clases posibles.
        umbral: valor por encima del cual se considera que la clase está presente.

    Returns:
        Lista de etiquetas presentes en el vector multihot.
    """
    return [clases[i] for i, v in enumerate(y) if v >= umbral]

# ============================================================
# 2. Audio → espectrograma
# ============================================================

import tensorflow as tf

MAX_SAMPLES = int(C.MAX_AUDIO_SECONDS * C.SAMPLE_RATE)  # 5s * 44100Hz = 220500 muestras

def cargar_audio(path):
    """Lee un .wav (PCM 16 bits, 44,1 kHz, mono) y devuelve la onda como
    tensor 1D float32 en [-1, 1]. No remuestrea (el dataset ya está a la frecuencia correcta).
    """    
    raw = tf.io.read_file(path)
    wav, = tf.audio.decode_wav(raw, desired_channels=1) # (n_samples, 1)
    return tf.squeeze(wav, axis=-1)  # (n_samples,)

def audio_a_logmel(wav):
    """Convierte una onda 1D en espectrograma log-mel (n_frames, N_MELS).
    Pesos: STFT (ventana 25ms / sal;to 10 ms) -> magnitud -> filtros mel.
    (96 bandas, 20 Hz-20 kHz) -> log (con offset para evitar log(0)).
    """
    stft = tf.signal.stft(
        wav,
        frame_length=C.WIN_LENGTH, # ≈ 1102 muestras (25 ms)
        frame_step=C.HOP_LENGTH, # ≈ 441  muestras (10 ms)
        fft_length=C.N_FFT, # 2048
    )  # (n_frames, n_fft//2+1)
    espectrograma = tf.abs(stft)  # magnitud
    matriz_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=C.N_MELS,
        num_spectrogram_bins=C.N_FFT // 2 + 1,
        sample_rate=C.SAMPLE_RATE,
        lower_edge_hertz=C.F_MIN,
        upper_edge_hertz=C.F_MAX,
    )
    mel = tf.matmul(espectrograma, matriz_mel)  # (n_frames, N_MELS)

    return tf.math.log(mel + C.LOG_OFFSET)  # log-mel (n_frames, N_MELS)

# ============================================================
# 3. Troceado en ventanas
# ============================================================

# constante derivada: cuantas columnas del espectrograma avanza cada ventana
WINDOW_HOP_FRAMES = int(round(C.WINDOW_HOP_SECONDS / C.STFT_HOP_SECONDS))  # 0.5/= 50 frames

def recortar(wav, training):
    """Acota la onda a MAX_AUDIO_SECONDS.
    - train: tramo aleatorio (funciona como data augmentation)
    - val/test: tramo inicial (determinista)
    clips mas cortos se rellenan con ceros al final (padding).
    """
    n = tf.shape(wav)[0]
    def _recortar():
        if training:
            inicio = tf.random.uniform([], 0, tf.maximum(1, n - MAX_SAMPLES), dtype=tf.int32)
        else:
            inicio = (n - MAX_SAMPLES) // 2 if n > MAX_SAMPLES else 0
        return wav[inicio:inicio + MAX_SAMPLES]
    return tf.cond(n > MAX_SAMPLES, _recortar, lambda: wav)

def trocear_en_ventanas(logmel):
    """(n_frames, N_MELS) → (n_windows, N_MELS, N_FRAMES, 1).
    trocea el espectrograma log-mel en ventanas de N_FRAMES columnas, con salto
    WINDOW_HOP_FRAMES. pad_end=True garantiza al menos 1 ventana (rellenando con ceros la última si no completa).
    """
    v = tf.signal.frame(
        logmel,
        frame_length=C.N_FRAMES,        # 100 columnas por ventana (= 1 s)
        frame_step=WINDOW_HOP_FRAMES,   # 50 columnas de salto (= 0.5 s)
        axis=0,
        pad_end=True,
    )                                    # (n_windows, N_FRAMES, N_MELS)
    v = tf.transpose(v, [0, 2, 1])       # (n_windows, N_MELS, N_FRAMES)
    return tf.expand_dims(v, axis=-1)    # (n_windows, N_MELS, N_FRAMES, 1)

# ============================================================
# 4. Batching con máscara
# ============================================================

from pathlib import Path

def split_train_val(df, val_split=None, seed=None):
    """Separa un 15% para validación. Estratifica por la etiqueta principal
    (primera de la lista) como proxy razonable en multi-etiqueta."""
    from sklearn.model_selection import train_test_split
    val_split = val_split or C.VAL_SPLIT
    seed = seed if seed is not None else C.SEED
    principal = df["labels"].str.split(",").str[0]      # proxy de estratificación
    try:
        df_tr, df_va = train_test_split(
            df, test_size=val_split, random_state=seed, stratify=principal)
    except ValueError:                                   # clases con muy pocas muestras
        df_tr, df_va = train_test_split(df, test_size=val_split, random_state=seed)
    return df_tr.reset_index(drop=True), df_va.reset_index(drop=True)

def _preparar_ejemplo(ruta, training):
    """Encadena carga -> recorte -> log-mel -> ventanas, y genera la máscara."""
    wav      = cargar_audio(ruta)
    wav      = recortar(wav, training)
    logmel   = audio_a_logmel(wav)
    # (parte 5) aquí irá SpecAugment si training y AUGMENTATION lo pide
    ventanas = trocear_en_ventanas(logmel)                       # (n_windows, 96, 100, 1)
    mascara  = tf.ones(tf.shape(ventanas)[0], dtype=tf.float32)  # (n_windows,)  todo 1 = reales
    return ventanas, mascara

def construir_dataset(df, clases, training, audio_dir=None):
    """tf.data.Dataset que entrega ((ventanas, máscara), etiquetas).
    - train: shuffle + recorte aleatorio (+ augmentation en parte 5).
    - val/test: determinista, sin augmentation."""
    audio_dir = Path(audio_dir) if audio_dir else (C.DATA_RAW / "train_curated")
    rutas = [str(audio_dir / f) for f in df["fname"].values]
    Y     = construir_matriz_etiquetas(df, clases)               # (n, 80) float32

    ds = tf.data.Dataset.from_tensor_slices((rutas, Y))
    if training:
        ds = ds.shuffle(len(rutas), seed=C.SEED, reshuffle_each_iteration=True)

    def _map(ruta, y):
        ventanas, mascara = _preparar_ejemplo(ruta, training)
        return (ventanas, mascara), y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

    # padded_batch: iguala n_windows dentro del batch y RELLENA con 0
    ds = ds.padded_batch(
        C.BATCH_SIZE,
        padded_shapes=(([None, C.N_MELS, C.N_FRAMES, 1], [None]), [C.NUM_CLASSES]),
        padding_values=((0.0, 0.0), 0.0),
    )
    return ds.prefetch(tf.data.AUTOTUNE)

# ============================================================
# 5. Augmentación (train only, según config.AUGMENTATION)
# ============================================================

def aplicar_specaugment(logmel):
    """Enmascara SPECAUG_TIME_MASKS bandas temporales y SPECAUG_FREQ_MASKS
    bandas de frecuencia (las pone en 0). Actúa sobre (n_frames, N_MELS)."""
    T = tf.shape(logmel)[0]              # frames (dinámico)
    M = C.N_MELS                         # 96 (fijo)
    ancho_t = tf.maximum(1, T // 8)      # ancho máx de banda temporal
    ancho_f = max(1, M // 8)             # ancho máx de banda de frecuencia

    def tapar(x, largo, ancho_max, eje):
        ancho  = tf.random.uniform([], 0, ancho_max + 1, dtype=tf.int32)
        inicio = tf.random.uniform([], 0, tf.maximum(1, largo - ancho), dtype=tf.int32)
        idx    = tf.range(largo)
        fuera  = tf.logical_or(idx < inicio, idx >= inicio + ancho)   # True = conservar
        keep   = tf.cast(fuera, x.dtype)
        keep   = keep[:, None] if eje == 0 else keep[None, :]         # difundir al otro eje
        return x * keep

    out = logmel
    for _ in range(C.SPECAUG_TIME_MASKS):
        out = tapar(out, T, ancho_t, eje=0)
    for _ in range(C.SPECAUG_FREQ_MASKS):
        out = tapar(out, M, ancho_f, eje=1)
    return out

def _muestrear_lambda(alpha):
    """lambda ~ Beta(alpha, alpha), usando dos Gamma (sin dependencias extra)."""
    g1 = tf.random.gamma([], alpha)
    g2 = tf.random.gamma([], alpha)
    return g1 / (g1 + g2)

def mixup_batch(inputs, y, alpha=None):
    """Mezcla lineal de ejemplos y etiquetas. Se aplica DESPUÉS de padded_batch
    (todos los ejemplos del batch ya tienen el mismo n_windows)."""
    alpha = alpha or C.MIXUP_ALPHA
    ventanas, mascara = inputs
    lam = _muestrear_lambda(alpha)
    idx = tf.random.shuffle(tf.range(tf.shape(ventanas)[0]))     # barajado del batch
    v2, m2, y2 = tf.gather(ventanas, idx), tf.gather(mascara, idx), tf.gather(y, idx)

    ventanas_mix = lam * ventanas + (1.0 - lam) * v2
    y_mix        = lam * y        + (1.0 - lam) * y2
    mascara_mix  = tf.maximum(mascara, m2)      # unión de ventanas reales
    return (ventanas_mix, mascara_mix), y_mix

# ============================================================
# 6. Ensamblado del dataset
# ============================================================

# ============================================================
# 7. (Opcional pero clave) Precómputo
# ============================================================

# ============================================================
# 8. Split
# ============================================================