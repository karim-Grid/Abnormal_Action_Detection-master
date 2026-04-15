"""Modèle C3D (Sports-1M) — extraction des activations FC7."""

from __future__ import annotations

import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import get_file
except ImportError as e:
    raise ImportError("TensorFlow est requis: pip install tensorflow") from e

WEIGHTS_URL = "https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5"
WEIGHTS_MD5 = "b7a93b2f9156ccbebe3ca24b41fc5402"


def build_c3d_base() -> keras.Model:
    """C3D jusqu'à FC7 (sans FC8), entrée (16, 112, 112, 3)."""
    shape0 = (16, 112, 112, 3)
    m = Sequential(name="c3d_base")
    m.add(
        layers.Conv3D(
            64, 3, activation="relu", padding="same", name="conv1", input_shape=shape0
        )
    )
    m.add(
        layers.MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding="same", name="pool1"
        )
    )
    m.add(layers.Conv3D(128, 3, activation="relu", padding="same", name="conv2"))
    m.add(
        layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool2"
        )
    )
    m.add(layers.Conv3D(256, 3, activation="relu", padding="same", name="conv3a"))
    m.add(layers.Conv3D(256, 3, activation="relu", padding="same", name="conv3b"))
    m.add(
        layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool3"
        )
    )
    m.add(layers.Conv3D(512, 3, activation="relu", padding="same", name="conv4a"))
    m.add(layers.Conv3D(512, 3, activation="relu", padding="same", name="conv4b"))
    m.add(
        layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool4"
        )
    )
    m.add(layers.Conv3D(512, 3, activation="relu", padding="same", name="conv5a"))
    m.add(layers.Conv3D(512, 3, activation="relu", padding="same", name="conv5b"))
    m.add(layers.ZeroPadding3D(padding=(0, 1, 1)))
    m.add(
        layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool5"
        )
    )
    m.add(layers.Flatten())
    m.add(layers.Dense(4096, activation="relu", name="fc6"))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(4096, activation="relu", name="fc7"))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(487, activation="softmax", name="fc8"))
    return m


def load_c3d_fc7(
    weights_path: str | None = None,
) -> keras.Model:
    """
    Charge les poids Sports-1M et renvoie un modèle dont la sortie est FC7.

    Si weights_path est fourni, ce fichier .h5 est utilisé (sinon téléchargement).
    """
    full = build_c3d_base()
    path = weights_path or get_file(
        "sports1M_weights_tf.h5",
        WEIGHTS_URL,
        cache_subdir="models",
        md5_hash=WEIGHTS_MD5,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    full.load_weights(path)
    # Modèle intermédiaire: entrée -> fc7
    out = full.get_layer("fc7").output
    fc7_model = keras.Model(inputs=full.input, outputs=out, name="c3d_fc7")
    return fc7_model
