"""Shared .keras model loader with Keras 3 serialisation workaround.

Keras 3 (TF 2.16+) writes extra config keys (renorm, quantization_config)
that it refuses to accept on load.  This module strips them from the JSON
before deserialization.  All affected values are just ``False`` / ``None``
so this is a safe no-op.
"""

import json
import os
import tempfile
import zipfile

import tensorflow as tf


_STRIP_KEYS = {"renorm", "renorm_clipping", "renorm_momentum", "quantization_config"}


def _walk(d):
    if isinstance(d, dict):
        if "config" in d:
            for k in _STRIP_KEYS:
                d["config"].pop(k, None)
        for v in d.values():
            _walk(v)
    elif isinstance(d, list):
        for v in d:
            _walk(v)


def load_keras_model(path: str | os.PathLike) -> tf.keras.Model:
    """Load a .keras model, stripping problematic Keras 3 config keys."""
    path = str(path)

    with zipfile.ZipFile(path) as z:
        entries = {name: z.read(name) for name in z.namelist()}
        config = json.loads(entries.pop("config.json"))
        _walk(config)

    tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, data in entries.items():
                zout.writestr(name, data)
            zout.writestr("config.json", json.dumps(config).encode("utf-8"))
        return tf.keras.models.load_model(tmp.name)
    finally:
        os.unlink(tmp.name)
