"""Détection légère sans deep learning : mouvement inter-frames et irrégularité locale (OpenCV + NumPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cv2

from aad.video_io import frame_windows, read_video_frames


@dataclass
class WindowScoreLite:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    anomaly_score: float
    feature_delta_l2: float


def _gray(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError("frames doit être (T,H,W,3)")
    out = np.zeros(frames.shape[:3], dtype=np.float32)
    for i in range(frames.shape[0]):
        out[i] = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sigma


def analyze_video_lite(
    video_path: str,
    *,
    window: int = 16,
    stride: int = 8,
    max_frames: int | None = None,
) -> tuple[list[WindowScoreLite], dict]:
    """
    Score par fenêtre : combinaison de l'énergie de mouvement moyenne et de sa variance
    (pics = changements brusques ou chaos local).
    """
    frames, fps = read_video_frames(video_path, max_frames=max_frames)
    n = frames.shape[0]
    g = _gray(frames)
    wins = frame_windows(n, window=window, stride=stride)

    motion_energy: list[float] = []
    motion_var: list[float] = []
    meta_ranges: list[tuple[int, int]] = []

    for start, end in wins:
        clip = g[start:end]
        if clip.shape[0] < 2:
            motion_energy.append(0.0)
            motion_var.append(0.0)
            meta_ranges.append((start, end))
            continue
        diffs = np.abs(np.diff(clip, axis=0))
        me = float(np.mean(diffs))
        mv = float(np.var(diffs))
        motion_energy.append(me)
        motion_var.append(mv)
        meta_ranges.append((start, end))

    me = np.array(motion_energy, dtype=np.float64)
    mv = np.array(motion_var, dtype=np.float64)
    combined = _zscore(me) + _zscore(mv)
    combined = np.maximum(combined, 0.0)

    scores: list[WindowScoreLite] = []
    for i, ((start, end), s) in enumerate(zip(meta_ranges, combined)):
        t0 = start / fps if fps > 0 else 0.0
        t1 = (end - 1) / fps if fps > 0 else float(end - start)
        scores.append(
            WindowScoreLite(
                start_frame=start,
                end_frame=end,
                start_sec=t0,
                end_sec=t1,
                anomaly_score=float(s),
                feature_delta_l2=float(me[i]),
            )
        )

    info = {
        "fps": fps,
        "num_frames": n,
        "num_windows": len(scores),
        "window": window,
        "stride": stride,
        "backend": "lite",
        "note": "Backend léger (mouvement) — installez TensorFlow (Python 3.10–3.12) pour C3D.",
    }
    return scores, info
