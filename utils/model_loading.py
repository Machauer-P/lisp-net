"""Shared .keras model loader with Keras 3 serialisation workaround.

Keras 3 (TF 2.16+) writes extra config keys (renorm, quantization_config)
that it refuses to accept on load.  This module strips them from the JSON
before deserialization.  All affected values are just ``False`` / ``None``
so this is a safe no-op.

Model paths that look like Hugging Face repo IDs (``user/repo``) are
automatically downloaded from the HF Hub and cached locally.

Loading works across Keras 2 (TF < 2.16) and Keras 3 (TF ≥ 2.16) by
building the architecture from config and then loading weights via the
official ``load_weights_only`` API — avoiding ``load_model`` on a patched
ZIP, which only works when Keras recognises ``.keras`` as a ZIP archive.
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import tensorflow as tf


# Keys that Keras 3 writes into layer configs but Keras 2 rejects.
_STRIP_KEYS = {"renorm", "renorm_clipping", "renorm_momentum", "quantization_config", "synchronized"}

# Keras 3 → 2 InputLayer renames (Keras 2 uses different names / lacks features).
_INPUT_LAYER_RENAMES = {"batch_shape": "batch_input_shape"}
_INPUT_LAYER_DROP = {"optional", "sparse", "ragged"}

# Keras 3 → 2 parameter renames for other layer types.
_LAYER_RENAMES = {
    "LeakyReLU": {"negative_slope": "alpha"},
}

# Keras 3 layer classes that don't exist in Keras 2 — mapped to
# (new_class_name, extra_config_entries).
_LAYER_CLASS_REPLACEMENTS = {
    "Sigmoid": ("Activation", {"activation": "sigmoid"}),
}


def _convert_inbound_args(args):
    """Convert Keras 3 inbound ``args`` to Keras 2 flat arg list.

    Keras 3 serializes a list-valued argument (e.g. for merge layers) as a
    single ``args`` entry — ``[[t_a, t_b]]``.  Keras 2 expects each tensor
    to be a separate entry — ``[t_a, t_b]``.  We unwrap one level so that
    ``process_node`` receives the correct number of positional arguments.
    """
    flat = []
    for arg in args:
        ref = _convert_tensor_ref(arg)
        if isinstance(arg, list) and not (isinstance(arg, dict) and
                                          arg.get("class_name") == "__keras_tensor__"):
            # K3 list-of-tensors argument → unwrap into individual K2 refs.
            if isinstance(ref, list):
                flat.extend(ref)
            else:
                flat.append(ref)
        else:
            flat.append(ref)
    return flat


def _convert_tensor_ref(obj):
    """Convert a ``__keras_tensor__`` dict to a ``keras_history`` list."""
    if isinstance(obj, dict) and obj.get("class_name") == "__keras_tensor__":
        return obj["config"]["keras_history"]
    if isinstance(obj, list):
        return [_convert_tensor_ref(o) for o in obj]
    return obj


def _patch_keras3_strip_only(d):
    """Lightweight config patch: only strip ``_STRIP_KEYS``.
    Used by the Keras 3 primary path where a patched ``.keras`` ZIP is
    fed to ``load_model()`` — Keras 3 understands the native format and
    only chokes on the unwanted BatchNorm / Dense keys.
    """
    if isinstance(d, dict):
        if "config" in d and isinstance(d["config"], dict):
            for k in _STRIP_KEYS:
                d["config"].pop(k, None)
        for v in d.values():
            _patch_keras3_strip_only(v)
    elif isinstance(d, list):
        for v in d:
            _patch_keras3_strip_only(v)


# Default Hugging Face repo
_DEFAULT_HF_REPO = "Machauer-P/lisp-net"

# Hugging Face cache directory
_HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lisp-net")


def _patch_keras3_config(d):
    """Recursively patch a Keras 3 *config.json* so Keras 2 can deserialise it.

    Transformations applied (all are safe no-ops on Keras 3):
    * Strip Keras 3 metadata (``registered_name``, ``shared_object_id``,
      ``build_config``, ``compile_config``).
    * Strip ``_STRIP_KEYS`` from every layer-config sub-dict.
    * Rename / drop ``InputLayer`` keys that Keras 2 does not recognise.
    * Replace Keras 3-only layer classes (e.g. ``Sigmoid`` → ``Activation``).
    * Rename per-layer parameters (e.g. ``negative_slope`` → ``alpha``).
    * Flatten nested ``DTypePolicy`` objects to plain dtype strings.
    """
    if isinstance(d, dict):
        # Flatten Keras 3 DTypePolicy → plain dtype string.
        if d.get("class_name") == "DTypePolicy" and "config" in d:
            return d["config"].get("name", "float32")

        # Remove Keras 3-only metadata that confuses Keras 2.
        d.pop("registered_name", None)
        d.pop("shared_object_id", None)
        d.pop("build_config", None)
        d.pop("compile_config", None)

        # Convert inbound_nodes from Keras 3 dict format to Keras 2 list format.
        # K3: [{"args": [__keras_tensor__], "kwargs": {}}, ...]
        # K2: [[["layer", 0, 0], ...], ...]
        if "inbound_nodes" in d and isinstance(d["inbound_nodes"], list):
            converted = []
            for node in d["inbound_nodes"]:
                if isinstance(node, dict):
                    # Keras 3 format — extract args, drop kwargs, flatten tensors.
                    args = node.get("args", [])
                    converted.append(_convert_inbound_args(args))
                else:
                    converted.append(node)
            d["inbound_nodes"] = converted

        # Patch layer "config" sub-dicts.
        if "config" in d and isinstance(d["config"], dict):
            cfg = d["config"]
            for k in _STRIP_KEYS:
                cfg.pop(k, None)

            # InputLayer compat
            if d.get("class_name") == "InputLayer":
                for old, new in _INPUT_LAYER_RENAMES.items():
                    if old in cfg:
                        cfg[new] = cfg.pop(old)
                for k in _INPUT_LAYER_DROP:
                    cfg.pop(k, None)

            # Replace Keras 3-only layer classes with Keras 2 equivalents.
            cls = d.get("class_name", "")
            if cls in _LAYER_CLASS_REPLACEMENTS:
                new_cls, extra_cfg = _LAYER_CLASS_REPLACEMENTS[cls]
                d["class_name"] = new_cls
                d["module"] = "keras.layers"
                cfg.update(extra_cfg)

            # Per-layer Keras 3 → 2 parameter renames.
            if cls in _LAYER_RENAMES:
                for old, new in _LAYER_RENAMES[cls].items():
                    if old in cfg:
                        cfg[new] = cfg.pop(old)

        return {k: _patch_keras3_config(v) for k, v in d.items()}

    if isinstance(d, list):
        return [_patch_keras3_config(v) for v in d]

    return d


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


def _resolve_model_path(model_ref: Optional[str]) -> str:
    """
    Resolve a model reference to a local ``.keras`` path.

    - ``None`` → download from the default Hugging Face repo.
    - ``user/repo`` → download from that HF repo.
    - local ``.keras`` path → used as-is.
    """
    if model_ref is None:
        return _hf_download(_DEFAULT_HF_REPO)

    p = Path(model_ref)

    # If the path exists on disk, use it directly — avoids false
    # positives on Windows where ``Path.as_posix()`` inserts "/"
    # that would otherwise be mistaken for an HF repo ID.
    if p.exists():
        return str(p)

    # If the path looks like a local file (has a model extension),
    # don't attempt HF download even if it doesn't exist — let the
    # caller get a clear "file not found" error.
    if p.suffix in (".keras", ".h5", ".hdf5"):
        return str(p)

    # HF repo IDs look like "user/repo" (single slash, no colon,
    # no file extension).  On Windows a drive-letter path like
    # "C:/..." would still contain "/" after as_posix(), so only
    # try the HF download when the path does NOT exist locally
    # and the reference is not an absolute / drive-letter path.
    if "/" in str(model_ref) and not os.path.isabs(str(model_ref)):
        return _hf_download(model_ref)

    return str(p)


def _load_weights_from_keras3_h5(model: tf.keras.Model, weights_data: bytes) -> None:
    """Load weights from a Keras 3 ``model.weights.h5`` byte blob.

    Uses the official ``load_weights_only`` API (Keras >= 3) which matches
    layers by class-based traversal naming — the same convention used when the
    HDF5 archive inside a ``.keras`` file is written.  Falls back to manual
    class-based matching when the API is unavailable (Keras 2).
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".weights.h5")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(weights_data)

        try:
            from keras.src.saving.saving_lib import load_weights_only
        except ImportError:
            try:
                from keras.saving.saving_lib import load_weights_only
            except ImportError:
                load_weights_only = None

        if load_weights_only is not None:
            load_weights_only(model, tmp_path)
        else:
            # Keras 2 fallback — match weights by class-based traversal names,
            # which are the keys used in the HDF5 layers group.
            import h5py
            from keras.src.utils import naming

            with h5py.File(tmp_path, "r") as h5f:
                layers_group = h5f["layers"]
                used_names = {}
                for layer in model.layers:
                    class_name = naming.to_snake_case(layer.__class__.__name__)
                    if class_name in used_names:
                        used_names[class_name] += 1
                        class_name = f"{class_name}_{used_names[class_name]}"
                    else:
                        used_names[class_name] = 0

                    if class_name not in layers_group:
                        continue
                    lg = layers_group[class_name]
                    if "vars" not in lg:
                        continue

                    vars_group = lg["vars"]
                    weights = [vars_group[str(i)][...] for i in range(len(vars_group))]
                    if weights:
                        layer.set_weights(weights)
    finally:
        os.unlink(tmp_path)


def load_keras_model(path: Optional[str] = None) -> tf.keras.Model:
    """Load a .keras model, stripping problematic Keras 3 config keys.

    Works across Keras 2 (TF < 2.16) and Keras 3 (TF ≥ 2.16).

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
    # Normalise the input to a plain string path.
    if isinstance(path, Path):
        path = path.as_posix()
    elif path is not None:
        path = str(path)
    path = _resolve_model_path(path)

    # Read the Keras-native ZIP archive.
    with zipfile.ZipFile(path) as z:
        entries = {name: z.read(name) for name in z.namelist()}
        raw_config = json.loads(entries.pop("config.json"))

    # --- Primary path: patched .keras ZIP → load_model (Keras 3 native).
    # Only strip the problematic BatchNorm/Dense keys; Keras 3 understands
    # the rest of the config format natively.
    config_v3 = json.loads(json.dumps(raw_config))  # deep copy
    _patch_keras3_strip_only(config_v3)

    tmp_keras = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    tmp_keras.close()
    try:
        with zipfile.ZipFile(tmp_keras.name, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, data in entries.items():
                zout.writestr(name, data)
            zout.writestr("config.json", json.dumps(config_v3).encode("utf-8"))

        model = tf.keras.models.load_model(tmp_keras.name)
        return model
    except (OSError, ValueError, AttributeError, TypeError):
        # Keras 2 fallback: load_model() on a .keras file fails because
        # Keras 2 expects a SavedModel directory, not a ZIP archive.
        pass
    finally:
        os.unlink(tmp_keras.name)

    # --- Keras 2 fallback: model_from_json with full config translation.
    config_v2 = _patch_keras3_config(raw_config)
    model = tf.keras.models.model_from_json(json.dumps(config_v2))

    # Locate and load the weights HDF5 file embedded in the archive.
    weights_key = None
    for name in entries:
        if name.endswith((".h5", ".hdf5")):
            weights_key = name
            break

    if weights_key is not None:
        _load_weights_from_keras3_h5(model, entries[weights_key])

    return model
