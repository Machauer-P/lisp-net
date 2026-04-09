import tensorflow as tf
import tensorflow_probability as tfp

def shaping(tensor, h=128, w=128, binary=False):
    """Ensure proper shape (1, 128, 128, 1/2) of tf tensor.
    """
    if len(tensor.shape) == 3:
        # (128,128,1/2)
        if tensor.shape[0] > 1 and tensor.shape[1] > 1 and (tensor.shape[2] == 1 or tensor.shape[2] == 2):
            tensor = tensor[tf.newaxis,...]
        # (1,128,128)
        elif tensor.shape[0] == 1 and tensor.shape[1] > 1 and tensor.shape[2] > 1:
            tensor = tensor[...,tf.newaxis]

    if len(tensor.shape) == 2:
        tensor = tensor[tf.newaxis,...,tf.newaxis]

    if tensor.shape[1] != h or tensor.shape[2] != w:
        if binary:
            tensor = tf.image.resize(tensor, [h, w], method='nearest')
        else:
            tensor = tf.image.resize(tensor, [h, w])

    if tensor.shape == (1,h,w,1) or tensor.shape == (1,h,w,2):
        pass
    else:
        raise Exception(f'Something went wrong. Shape is {tensor.shape}.')

    return tensor

def min_max_norm(image, lower_q=0.5, upper_q=99.5):
    """Robust min-max normalization using quantiles.
    """
    image = tf.cast(image, tf.float32)
    flat = tf.reshape(image, [-1])

    # Berechne Quantile
    q_min = tfp.stats.percentile(flat, lower_q, interpolation='nearest')
    q_max = tfp.stats.percentile(flat, upper_q, interpolation='nearest')

    # Clip nur innerhalb der Quantile (robust)
    image = tf.clip_by_value(image, q_min, q_max)

    # Min–Max Normalisierung
    image = (image - q_min) / (q_max - q_min + 1e-8)
    return image
