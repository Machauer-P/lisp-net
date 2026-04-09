"""
ds_handler_2d.py
===============
Utility functions for saving and loading 2D datasets using the TFRecord format.
This module is specifically used for 2D evaluation benchmarks and comparisons
against models like UniverSeg.

Standard Dataset Element:
------------------------
A tuple (x, y, p) where:
    - x: Input image tensor
    - y: Segmentation mask tensor
    - p: Prompt/Context data tensor (e.g., support images and masks for UniverSeg)

The data is stored as serialized tensors in TFRecord files to preserve 
precision and structural information.
"""

import os
import tensorflow as tf


def save_tf_dataset_2D(tf_ds, filename, path=".", batch_size=32):
    """
    Save a tf.data.Dataset of (x, y), (x, y, p), or (x, y, p, offset) tuples to a TFRecord file.

    Parameters:
    -----------
    tf_ds : tf.data.Dataset
        The dataset to save.
    filename : str
        The name of the TFRecord file to create.
    path : str, optional
        The directory where the file will be saved. Defaults to ".".
    batch_size : int, optional
        The batch size for processing the dataset during save. Defaults to 32.
    """
    os.makedirs(path, exist_ok=True)
    if not filename.endswith(".tfrecord"):
        filename += ".tfrecord"
    filepath = os.path.join(path, filename)

    # Peek at the first element to determine structure
    try:
        sample = next(iter(tf_ds.take(1)))
        num_elements = len(sample)
    except Exception:
        num_elements = 0

    def serialize_example(elements):
        feature = {
            'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(elements[0]).numpy()])),
            'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(elements[1]).numpy()])),
        }
        if len(elements) >= 3:
            feature['p'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(elements[2]).numpy()]))
        if len(elements) >= 4:
            feature['offset'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(elements[3]).numpy()]))
        
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    tf_ds_batched = tf_ds.batch(batch_size)

    with tf.io.TFRecordWriter(filepath) as writer:
        for batch in tf_ds_batched:
            # batch is a tuple of batched tensors: (x_batch, y_batch, ...)
            # We want to iterate through the batch dimension (axis 0)
            rows = zip(*batch) if isinstance(batch, (tuple, list)) else batch
            for row in rows:
                writer.write(serialize_example(row))


def load_tf_dataset_2D(filename, path=".", include_offset=False, include_prompt=True):
    """
    Load a dataset from a TFRecord file.

    Parameters:
    -----------
    filename : str
    path : str
    include_offset : bool
        Whether to include the 'offset' feature.
    include_prompt : bool
        Whether to include the 'p' (prompt) feature.

    Returns:
    --------
    tf.data.Dataset
        Yields (x, y), (x, y, p), or (x, y, p, offset) depending on arguments.
    """
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

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        x = tf.io.parse_tensor(parsed['x'], out_type=tf.float32)
        y = tf.io.parse_tensor(parsed['y'], out_type=tf.float32)
        
        results = [x, y]
        if include_prompt:
            p = tf.io.parse_tensor(parsed['p'], out_type=tf.float32)
            results.append(p)
        if include_offset:
            offset = tf.io.parse_tensor(parsed['offset'], out_type=tf.int32)
            results.append(tf.cast(offset, tf.float32))
            
        return tuple(results)

    return tf.data.TFRecordDataset(filepath).map(_parse_function)