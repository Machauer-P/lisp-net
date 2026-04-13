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

def unified_z_score_norm(volume):
    """
    Robust 3D Z-Score normalization that blindly handles mixed CT and MRI datasets.
    Calculates statistics only on the foreground to prevent background-skew.
    """
    volume = tf.cast(volume, tf.float32)

    # 1. Blindly determine if CT or MRI
    # CTs have negative values (air is ~ -1000). MRIs are strictly >= 0.
    v_min = tf.reduce_min(volume)
    is_ct = tf.cast(tf.less(v_min, -100.0), tf.float32) 
    
    # Set background threshold: Ignore < -500 for CT, and < 1e-3 for MRI
    threshold = is_ct * (-500.0) + (1.0 - is_ct) * 1e-3
    
    # 2. Isolate foreground (patient tissue) pixels
    foreground_mask = tf.greater(volume, threshold)
    foreground_pixels = tf.boolean_mask(volume, foreground_mask)
    
    # Edge case fallback: If the volume is entirely background, use all pixels
    foreground_pixels = tf.cond(
        tf.equal(tf.size(foreground_pixels), 0),
        lambda: tf.reshape(volume, [-1]),
        lambda: foreground_pixels
    )

    # 3. Calculate Mean and Standard Deviation ONLY on the foreground tissue
    mean = tf.reduce_mean(foreground_pixels)
    std = tf.math.reduce_std(foreground_pixels)

    # Prevent division by zero
    std = tf.maximum(std, 1e-8)

    # 4. Apply Z-Score normalization to the ENTIRE volume
    normalized_volume = (volume - mean) / std
    
    # 5. Clip extreme outliers (e.g., metal artifacts in CT or noise spikes in MRI)
    normalized_volume = tf.clip_by_value(normalized_volume, -5.0, 5.0)

    return normalized_volume


def universal_normalization(volume, modality="CT"):
    """
    Unified intensity normalization for CT, MRI, and any other modality.

    Workflow (applied in this order):
        1. Clip    — removes extreme outliers / artifacts
        2. Normalize — z-score using either hardcoded (CT) or adaptive (MRI) stats

    CT strategy
    -----------
    Uses *hardcoded* global statistics derived from large multi-organ CT datasets.
    This is valid because CT intensity is physically calibrated in Hounsfield Units
    (HU), which are scanner-independent.  Clipping to [-1000, 1000] covers the
    full range of soft tissue, bone, and air.

    MRI / Other strategy
    --------------------
    MRI intensity has no physical unit and varies arbitrarily between scanners,
    field strengths, and pulse sequences.  Per-volume statistics are therefore
    *required* for cross-scanner generalisation.  Foreground is isolated before
    computing statistics to prevent the large background region from biasing the
    mean/std towards zero.

    Args:
        volume   : tf.Tensor or np.ndarray of shape (Z, Y, X).
        modality : str — "CT" for computed tomography, anything else for MRI/other.

    Returns:
        tf.Tensor (float32), values clipped to [-5.0, 5.0].
    """
    volume = tf.cast(volume, tf.float32)

    if modality == "CT":
        # 1. Clip to broad body window — removes metal artifacts and air outliers.
        volume = tf.clip_by_value(volume, -1000.0, 1000.0)

        # 2. Apply fixed global z-score statistics (physically meaningful for HU).
        #    mean ≈ -15 HU, std ≈ 160 HU are representative of a general body window.
        global_mean = -15.0
        global_std  = 160.0
        normalized_volume = (volume - global_mean) / global_std

    else:  # MRI / Other
        # 1. Compute foreground mask *before* clipping so that background zeros
        #    are always excluded regardless of the percentile result.
        foreground_mask = tf.greater(volume, 0.0)
        foreground_pixels = tf.boolean_mask(volume, foreground_mask)

        # Fallback: if the volume is entirely zero, use all pixels.
        foreground_pixels = tf.cond(
            tf.equal(tf.size(foreground_pixels), 0),
            lambda: tf.reshape(volume, [-1]),
            lambda: foreground_pixels,
        )

        # 2. Percentile clipping — only on foreground voxels.
        #    IMPORTANT: We must NOT apply tf.clip_by_value to the whole volume
        #    because that would raise background zeros up to p05 (a positive
        #    foreground value), making all background bright after z-scoring.
        p05  = tfp.stats.percentile(foreground_pixels, 0.5,  interpolation='nearest')
        p995 = tfp.stats.percentile(foreground_pixels, 99.5, interpolation='nearest')

        # Clip only where foreground is True; leave background at 0.
        clipped_values = tf.clip_by_value(volume, p05, p995)
        volume = tf.where(foreground_mask, clipped_values, volume)

        # 3. Masked z-score: compute stats on clipped foreground only.
        fg_clipped = tf.boolean_mask(volume, foreground_mask)
        fg_clipped = tf.cond(
            tf.equal(tf.size(fg_clipped), 0),
            lambda: tf.reshape(volume, [-1]),
            lambda: fg_clipped,
        )

        mean = tf.reduce_mean(fg_clipped)
        std  = tf.math.reduce_std(fg_clipped)

        # Z-score the entire volume, then force background to -5
        # so it renders as black and doesn't bias the model.
        normalized_volume = (volume - mean) / (std + 1e-8)
        normalized_volume = tf.where(foreground_mask, normalized_volume,
                                     tf.constant(-5.0, dtype=tf.float32))

    # Final symmetric clip to suppress any remaining outliers.
    return tf.clip_by_value(normalized_volume, -5.0, 5.0)