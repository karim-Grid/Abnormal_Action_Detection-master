"""Détection d'anomalies : C3D si TensorFlow disponible, sinon backend léger."""

from __future__ import annotations

import os

from aad.detector_lite import WindowScoreLite, analyze_video_lite

_FORCE = os.environ.get("AAD_BACKEND", "").lower()


def _tf_available() -> bool:
    if _FORCE == "lite":
        return False
    if _FORCE == "c3d" or _FORCE == "tf":
        return True
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


def analyze_video(
    video_path: str,
    *,
    window: int = 16,
    stride: int = 8,
    max_frames: int | None = None,
    weights_path: str | None = None,
):
    """
    Analyse une vidéo. Utilise les descripteurs C3D (FC7) si TensorFlow est installé,
    sinon un score basé sur le mouvement (OpenCV).
    """
    if _FORCE == "lite":
        return analyze_video_lite(
            video_path,
            window=window,
            stride=stride,
            max_frames=max_frames,
        )
    want_c3d = _tf_available()
    if want_c3d:
        try:
            from aad.detector_tf import analyze_video_tf

            return analyze_video_tf(
                video_path,
                window=window,
                stride=stride,
                max_frames=max_frames,
                weights_path=weights_path,
            )
        except ImportError:
            if _FORCE in ("c3d", "tf"):
                raise
    return analyze_video_lite(
        video_path,
        window=window,
        stride=stride,
        max_frames=max_frames,
    )


# Réexport pour le typage
__all__ = ["analyze_video", "WindowScoreLite"]
