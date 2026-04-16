"""
ds_handler_2d.py
================
Utility functions for saving and loading 2D evaluation datasets.
Used for few-shot segmentation benchmarks (e.g., UniverSeg comparison).

Storage format: NPZ bundle
--------------------------
Each dataset is stored as a single compressed NPZ file containing both
the query set and the support set:

    Query arrays  (N samples):
      x          (N, 128, 128, 1)  float32  — images, z-score normalised to [-5, 5]
      y          (N, 128, 128, 1)  float32  — binary segmentation labels
      p          (N, 128, 128, 2)  float32  — prompts [ref_image | ref_label]
      offset     (N,)              int32    — signed slice offset used to build the pair
      modality   (N,)              float32  — 0.0 = CT, 1.0 = MRI

    Support arrays  (S samples, default S=16):
      sx         (S, 128, 128, 1)  float32  — support images
      sy         (S, 128, 128, 1)  float32  — support labels
      s_modality (S,)              float32  — 0.0 = CT, 1.0 = MRI

Note on normalization
---------------------
Images (x, sx, and the first channel of p) are stored in the z-score
normalised range [-5, 5] — the training range of p_unet_292.

At inference time, models that require a different input range must apply
their own renormalisation:
  • Prompt-UNet  → use as-is  (trained on [-5, 5])
  • UniverSeg    → apply (x + 5) / 10  →  [0, 1]
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# NPZ bundle I/O  (current format)
# ---------------------------------------------------------------------------

def save_2d_npz_bundle(query_data, support_data, filename, path="."):
    """Save a 2D evaluation dataset as a single compressed NPZ bundle.

    Parameters
    ----------
    query_data   : dict with keys:
                     'x'        (N, 128, 128, 1) float32
                     'y'        (N, 128, 128, 1) float32
                     'p'        (N, 128, 128, 2) float32
                     'offset'   array-like int
                     'modality' array-like float32  (0=CT, 1=MRI)
    support_data : dict with keys:
                     'sx'         (S, 128, 128, 1) float32
                     'sy'         (S, 128, 128, 1) float32
                     's_modality' array-like float32  (0=CT, 1=MRI)
    filename     : str  — base name without extension
    path         : str  — output directory (created if missing)
    """
    os.makedirs(path, exist_ok=True)
    if filename.endswith('.npz'):
        filename = filename[:-4]
    filepath = os.path.join(path, filename)

    np.savez_compressed(
        filepath,
        # Query
        x        = np.asarray(query_data['x'],        dtype=np.float32),
        y        = np.asarray(query_data['y'],        dtype=np.float32),
        p        = np.asarray(query_data['p'],        dtype=np.float32),
        offset   = np.asarray(query_data['offset'],   dtype=np.int32),
        modality = np.asarray(query_data['modality'], dtype=np.float32),
        # Support
        sx         = np.asarray(support_data['sx'],         dtype=np.float32),
        sy         = np.asarray(support_data['sy'],         dtype=np.float32),
        s_modality = np.asarray(support_data['s_modality'], dtype=np.float32),
    )
    print(f"  Saved: {filepath}.npz")


def load_2d_npz_bundle(filename, path="."):
    """Load a 2D evaluation dataset from an NPZ bundle.

    Parameters
    ----------
    filename : str  — base name with or without '.npz'
    path     : str  — directory containing the file

    Returns
    -------
    query_data   : dict  {'x', 'y', 'p', 'offset', 'modality'}
    support_data : dict  {'sx', 'sy', 's_modality'}
    """
    if not filename.endswith('.npz'):
        filename += '.npz'
    filepath = os.path.join(path, filename)

    bundle = np.load(filepath)

    query_data = {
        'x':        bundle['x'],
        'y':        bundle['y'],
        'p':        bundle['p'],
        'offset':   bundle['offset'],
        'modality': bundle['modality'],
    }
    support_data = {
        'sx':         bundle['sx'],
        'sy':         bundle['sy'],
        's_modality': bundle['s_modality'],
    }
    return query_data, support_data


# ---------------------------------------------------------------------------
# Legacy TFRecord I/O  (kept for reference only — do not use for new data)
# ---------------------------------------------------------------------------

def save_tf_dataset_2D(tf_ds, filename, path=".", batch_size=32):
    """[DEPRECATED] Save a tf.data.Dataset to a TFRecord file.
    Use save_2d_npz_bundle() for new datasets.
    """
    import tensorflow as tf
    os.makedirs(path, exist_ok=True)
    if not filename.endswith(".tfrecord"):
        filename += ".tfrecord"
    filepath = os.path.join(path, filename)

    try:
        sample = next(iter(tf_ds.take(1)))
        num_elements = len(sample)
    except Exception:
        num_elements = 0

    def serialize_example(elements):
        feature = {
            'x': tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(elements[0]).numpy()])),
            'y': tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(elements[1]).numpy()])),
        }
        if len(elements) >= 3:
            feature['p'] = tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(elements[2]).numpy()]))
        if len(elements) >= 4:
            feature['offset'] = tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(elements[3]).numpy()]))
        proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return proto.SerializeToString()

    with tf.io.TFRecordWriter(filepath) as writer:
        for batch in tf_ds.batch(batch_size):
            rows = zip(*batch) if isinstance(batch, (tuple, list)) else batch
            for row in rows:
                writer.write(serialize_example(row))


def load_tf_dataset_2D(filename, path=".", include_offset=False, include_prompt=True):
    """[DEPRECATED] Load a dataset from a TFRecord file.
    Use load_2d_npz_bundle() for new datasets.
    """
    import tensorflow as tf
    if not filename.endswith(".tfrecord"):
        filename += ".tfrecord"
    filepath = os.path.join(path, filename)

    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    if include_prompt:
        feature_description['p'] = tf.io.FixedLenFeature([], tf.string)
    if include_offset:
        feature_description['offset'] = tf.io.FixedLenFeature([], tf.string)

    def _parse(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        x = tf.io.parse_tensor(parsed['x'], out_type=tf.float32)
        y = tf.io.parse_tensor(parsed['y'], out_type=tf.float32)
        results = [x, y]
        if include_prompt:
            results.append(tf.io.parse_tensor(parsed['p'], out_type=tf.float32))
        if include_offset:
            offset = tf.io.parse_tensor(parsed['offset'], out_type=tf.int32)
            results.append(tf.cast(offset, tf.float32))
        return tuple(results)

    return tf.data.TFRecordDataset(filepath).map(_parse)