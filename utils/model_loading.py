"""Shared .keras model loader with Keras 3 serialisation workaround.

Keras 3 (TF 2.16+) writes extra config keys (renorm, quantization_config)
that it refuses to accept on load.  This module strips them from the JSON
before deserialization.  All affected values are just ``False`` / ``None``
so this is a safe no-op.

Model paths that look like Hugging Face repo IDs (``user/repo``) are
automatically downloaded from the HF Hub and cached locally.
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import tensorflow as tf


_STRIP_KEYS = {"renorm", "renorm_clipping", "renorm_momentum", "quantization_config"}

# Default Hugging Face repo
_DEFAULT_HF_REPO = "Machauer-P/lisp-net"

# Hugging Face cache directory
_HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lisp-net")


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


def _hf_download(repo_id: str) -> str:
    """Download the .keras model from a Hugging Face repo, return local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models from Hugging Face. "
            "Install it with: pip install huggingface_hub"
        )

    os.makedirs(_HF_CACHE, exist_ok=True)

    return hf_hub_download(
        repo_id=repo_id,
        filename="lisp_net_332.keras",
        cache_dir=_HF_CACHE,
    )


def _resolve_model_path(model_ref: str | None) -> str:
    """
    Resolve a model reference to a local ``.keras`` path.

    - ``None`` → download from the default Hugging Face repo.
    - ``user/repo`` → download from that HF repo.
    - local ``.keras`` path → used as-is.
    """
    if model_ref is None:
        return _hf_download(_DEFAULT_HF_REPO)

    # Check for HF repo ID before Path conversion —
    # Path normalises slashes on Windows, which would break "/" detection.
    if "/" in model_ref:
        return _hf_download(model_ref)

    p = Path(model_ref)
    if p.suffix == ".keras" and p.exists():
        return str(p)

    return str(p)


def load_keras_model(path: str | os.PathLike | None = None) -> tf.keras.Model:
    """Load a .keras model, stripping problematic Keras 3 config keys.

    Parameters
    ----------
    path : str, Path, or None
        - ``None`` (default) — downloads from Hugging Face
          (``Machauer-P/lisp-net``).
        - ``user/repo`` — downloads from that HF repo.
        - local ``.keras`` file — loaded from disk.

    Returns
    -------
    tf.keras.Model
    """
    if isinstance(path, Path):
        path = path.as_posix()
    elif path is not None:
        path = str(path)
    path = _resolve_model_path(path)

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
