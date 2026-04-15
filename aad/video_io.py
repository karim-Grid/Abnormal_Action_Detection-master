"""Lecture vidéo et échantillonnage de fenêtres de frames."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV est requis: pip install opencv-python-headless") from e


def read_video_frames(path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """
    Charge toutes les frames RGB de la vidéo.

    Returns:
        frames: (T, H, W, 3) uint8
        fps: images par seconde (0 si inconnu)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames_list: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_list.append(rgb)
        if max_frames is not None and len(frames_list) >= max_frames:
            break
    cap.release()

    if not frames_list:
        raise ValueError("Aucune frame lue (fichier vide ou codec non supporté).")

    return np.stack(frames_list, axis=0), fps


def frame_windows(num_frames: int, window: int = 16, stride: int = 8) -> list[tuple[int, int]]:
    """Indices [start, end) pour chaque fenêtre (end exclusif)."""
    if num_frames < window:
        return [(0, num_frames)]

    out: list[tuple[int, int]] = []
    start = 0
    while start + window <= num_frames:
        out.append((start, start + window))
        start += stride
    if not out or out[-1][1] < num_frames:
        out.append((max(0, num_frames - window), num_frames))
    return out
