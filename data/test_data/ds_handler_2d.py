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
      x          (N, 128, 128, 1)  float32  — images, z-score [-5, 5]   (Prompt-UNet)
      x_u        (N, 128, 128, 1)  float32  — images, min-max [0, 1]    (UniverSeg)
      y          (N, 128, 128, 1)  float32  — binary segmentation labels (shared)
      p          (N, 128, 128, 2)  float32  — prompts [ref_image | ref_label]  (Prompt-UNet)
      offset     (N,)              int32    — signed slice offset used to build the pair
      modality   (N,)              float32  — 0.0 = CT, 1.0 = MRI

    Support arrays  (S samples, default S=16):
      sx         (S, 128, 128, 1)  float32  — images, z-score [-5, 5]   (Prompt-UNet)
      sx_u       (S, 128, 128, 1)  float32  — images, min-max [0, 1]    (UniverSeg)
      sy         (S, 128, 128, 1)  float32  — binary support labels       (shared)
      s_modality (S,)              float32  — 0.0 = CT, 1.0 = MRI

Normalization convention
------------------------
  • Prompt-UNet  → uses x  / p[..., 0]  (z-score, trained on [-5, 5])
  • UniverSeg    → uses x_u / sx_u      (per-volume min-max to [0, 1],
                                          matching UniverSeg's training pipeline)

  Both normalizations are applied at BUNDLE GENERATION TIME from the raw
  3-D volume, so cross-slice relative brightness is always preserved.
  No additional renormalization is needed at inference.

Backward compatibility
----------------------
  load_2d_npz_bundle() returns None for 'x_u' and 'sx_u' when loading
  bundles generated before this schema update (pre-April 2026).
  Re-generate the bundles with generate_2d_test_data.py to add these keys.
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
                     'x'        (N, 128, 128, 1) float32  z-score [-5,5]  (Prompt-UNet)
                     'x_u'      (N, 128, 128, 1) float32  min-max [0,1]   (UniverSeg)
                     'y'        (N, 128, 128, 1) float32  binary labels
                     'p'        (N, 128, 128, 2) float32  prompts
                     'offset'   array-like int
                     'modality' array-like float32  (0=CT, 1=MRI)
    support_data : dict with keys:
                     'sx'         (S, 128, 128, 1) float32  z-score [-5,5]  (Prompt-UNet)
                     'sx_u'       (S, 128, 128, 1) float32  min-max [0,1]   (UniverSeg)
                     'sy'         (S, 128, 128, 1) float32  binary labels
                     's_modality' array-like float32  (0=CT, 1=MRI)
    filename     : str  — base name without extension
    path         : str  — output directory (created if missing)
    """
    os.makedirs(path, exist_ok=True)
    if filename.endswith('.npz'):
        filename = filename[:-4]
    filepath = os.path.join(path, filename)

    kwargs = dict(
        # Query — z-score (Prompt-UNet)
        x        = np.asarray(query_data['x']),
        # Query — UniverSeg normalisation
        x_u      = np.asarray(query_data['x_u']),
        # Query — shared
        y        = np.asarray(query_data['y']),
        p        = np.asarray(query_data['p']),
        offset   = np.asarray(query_data['offset'],   dtype=np.int32),
        modality = np.asarray(query_data['modality'], dtype=np.float32),
        # Support — z-score (Prompt-UNet)
        sx         = np.asarray(support_data['sx']),
        # Support — UniverSeg normalisation
        sx_u       = np.asarray(support_data['sx_u']),
        # Support — shared
        sy         = np.asarray(support_data['sy']),
        s_modality = np.asarray(support_data['s_modality'], dtype=np.float32)
    )
    if 'task' in query_data:
        kwargs['task'] = np.asarray(query_data['task'], dtype=np.int32)
        
    np.savez_compressed(filepath, **kwargs)
    print(f"  Saved: {filepath}.npz")


def load_2d_npz_bundle(filename, path="."):
    """Load a 2D evaluation dataset from an NPZ bundle.

    Parameters
    ----------
    filename : str  — base name with or without '.npz'
    path     : str  — directory containing the file

    Returns
    -------
    query_data   : dict  {'x', 'x_u', 'y', 'p', 'offset', 'modality'}
    support_data : dict  {'sx_u', 'sy', 's_modality'}

    Note: 'x_u', 'sx' and 'sx_u' are None for bundles generated before the
    April 2026 schema update.  Re-generate with generate_2d_test_data.py.
    """
    if not filename.endswith('.npz'):
        filename += '.npz'
    filepath = os.path.join(path, filename)

    bundle = np.load(filepath, allow_pickle=True)

    query_data = {
        'x':        bundle['x'],
        'x_u':      bundle['x_u']  if 'x_u'  in bundle else None,
        'y':        bundle['y'],
        'p':        bundle['p'],
        'offset':   bundle['offset'],
        'modality': bundle['modality'],
        'task':     bundle['task'] if 'task' in bundle else None,
    }
    support_data = {
        'sx':         bundle['sx']   if 'sx'   in bundle else None,
        'sx_u':       bundle['sx_u'] if 'sx_u' in bundle else None,
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