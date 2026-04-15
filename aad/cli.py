"""Ligne de commande pour analyser une vidéo."""

from __future__ import annotations

import argparse
import json
import sys

from aad.detector import analyze_video


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Détection d'anomalies : C3D si TensorFlow est disponible, sinon scores de mouvement (OpenCV)."
    )
    p.add_argument("video", help="Chemin vers le fichier vidéo")
    p.add_argument(
        "-o",
        "--output",
        help="Fichier JSON de sortie (sinon stdout)",
    )
    p.add_argument("--window", type=int, default=16, help="Taille de fenêtre en frames")
    p.add_argument("--stride", type=int, default=8, help="Pas entre fenêtres")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limiter le nombre de frames (tests rapides)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Chemin optionnel vers sports1M_weights_tf.h5",
    )
    args = p.parse_args(argv)

    scores, meta = analyze_video(
        args.video,
        window=args.window,
        stride=args.stride,
        max_frames=args.max_frames,
        weights_path=args.weights,
    )
    out = {
        "meta": meta,
        "windows": [
            {
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "anomaly_score": s.anomaly_score,
                "feature_delta_l2": s.feature_delta_l2,
            }
            for s in scores
        ],
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
