"""Prétraitement C3D (redimensionnement, moyenne Sports-1M, crop)."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV est requis: pip install opencv-python-headless") from e

try:
    from tensorflow.keras.utils import get_file
except ImportError as e:
    raise ImportError("TensorFlow est requis: pip install tensorflow") from e

C3D_MEAN_URL = "https://github.com/adamcasson/c3d/releases/download/v0.1/c3d_mean.npy"
C3D_MEAN_MD5 = "08a07d9761e76097985124d9e8b2fe34"


def _ensure_mean_path() -> str:
    return get_file(
        "c3d_mean.npy",
        C3D_MEAN_URL,
        cache_subdir="models",
        md5_hash=C3D_MEAN_MD5,
    )


def preprocess_clip(video: np.ndarray) -> np.ndarray:
    """
    Prépare un clip pour C3D (channels_last).

    Args:
        video: (T, H, W, 3) uint8 ou float

    Returns:
        Tensor batch (1, 16, 112, 112, 3) float32
    """
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("video doit avoir la forme (T, H, W, 3)")

    t = video.shape[0]
    indices = np.ceil(np.linspace(0, t - 1, 16)).astype(int)
    frames = video[indices].astype(np.float32)

    reshaped = np.zeros((16, 128, 171, 3), dtype=np.float32)
    for i in range(16):
        reshaped[i] = cv2.resize(frames[i], (171, 128), interpolation=cv2.INTER_CUBIC)

    mean_path = _ensure_mean_path()
    mean = np.load(mean_path).astype(np.float32)
    reshaped -= mean
    cropped = reshaped[:, 8:120, 30:142, :]
    return np.expand_dims(cropped, axis=0)
