"""Pipeline C3D + FC7 (TensorFlow requis)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aad.c3d_model import load_c3d_fc7
from aad.preprocess import preprocess_clip
from aad.video_io import frame_windows, read_video_frames


@dataclass
class WindowScore:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    anomaly_score: float
    feature_delta_l2: float


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sigma


def analyze_video_tf(
    video_path: str,
    *,
    window: int = 16,
    stride: int = 8,
    max_frames: int | None = None,
    weights_path: str | None = None,
) -> tuple[list[WindowScore], dict]:
    frames, fps = read_video_frames(video_path, max_frames=max_frames)
    n = frames.shape[0]
    wins = frame_windows(n, window=window, stride=stride)

    model = load_c3d_fc7(weights_path=weights_path)
    features: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []

    for start, end in wins:
        clip = frames[start:end]
        if clip.shape[0] < window:
            pad = np.repeat(clip[-1:], window - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        batch = preprocess_clip(clip)
        feat = model.predict(batch, verbose=0)[0]
        features.append(feat.astype(np.float64))
        meta.append((start, end))

    if len(features) == 0:
        return [], {"fps": fps, "num_frames": n, "error": "no_windows", "backend": "c3d"}

    F = np.stack(features, axis=0)
    if len(F) > 1:
        deltas = np.linalg.norm(np.diff(F, axis=0), axis=1)
        delta_pad = np.concatenate([[0.0], deltas])
    else:
        delta_pad = np.zeros(1)
    dist_mean = np.linalg.norm(F - F.mean(axis=0, keepdims=True), axis=1)

    combined = _zscore(delta_pad) + _zscore(dist_mean)
    combined = np.maximum(combined, 0.0)

    scores: list[WindowScore] = []
    for i, ((start, end), s) in enumerate(zip(meta, combined)):
        t0 = start / fps if fps > 0 else 0.0
        t1 = (end - 1) / fps if fps > 0 else float(end - start)
        dval = float(delta_pad[min(i, len(delta_pad) - 1)])
        scores.append(
            WindowScore(
                start_frame=start,
                end_frame=end,
                start_sec=t0,
                end_sec=t1,
                anomaly_score=float(s),
                feature_delta_l2=dval,
            )
        )

    info = {
        "fps": fps,
        "num_frames": n,
        "num_windows": len(scores),
        "window": window,
        "stride": stride,
        "backend": "c3d",
    }
    return scores, info
