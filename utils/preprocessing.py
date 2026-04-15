import numpy as np

def universal_normalization(volume, modality="CT"):
    """
    Unified intensity normalization for CT, MRI, and any other modality.
    Pure numpy — no TF dependency.

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
        volume   : np.ndarray of shape (Z, Y, X) or any shape.
        modality : str — "CT" for computed tomography, anything else for MRI/other.

    Returns:
        np.ndarray (float32), values clipped to [-5.0, 5.0].
    """
    volume = np.asarray(volume, dtype=np.float32)

    if modality == "CT":
        # 1. Clip to broad body window — removes metal artifacts and air outliers.
        volume = np.clip(volume, -1000.0, 1000.0)

        # 2. Apply fixed global z-score statistics (physically meaningful for HU).
        #    mean ≈ -15 HU, std ≈ 160 HU are representative of a general body window.
        global_mean = -15.0
        global_std  = 160.0
        normalized_volume = (volume - global_mean) / global_std

    else:  # MRI / Other
        # 1. Compute foreground mask *before* clipping so that background zeros
        #    are always excluded regardless of the percentile result.
        foreground_mask = volume > 0.0
        foreground_pixels = volume[foreground_mask]

        # Fallback: if the volume is entirely zero, use all pixels.
        if foreground_pixels.size == 0:
            foreground_pixels = volume.reshape(-1)

        # 2. Percentile clipping — only on foreground voxels.
        #    IMPORTANT: We must NOT clip the whole volume because that would
        #    raise background zeros up to p05, making all background bright.
        # method='nearest' matches the original tfp.stats.percentile(interpolation='nearest')
        p05  = np.percentile(foreground_pixels, 0.5,  method='nearest')
        p995 = np.percentile(foreground_pixels, 99.5, method='nearest')

        # Clip only where foreground is True; leave background at its original value.
        clipped_values = np.clip(volume, p05, p995)
        volume = np.where(foreground_mask, clipped_values, volume)

        # 3. Masked z-score: compute stats on clipped foreground only.
        fg_clipped = volume[foreground_mask]
        if fg_clipped.size == 0:
            fg_clipped = volume.reshape(-1)

        mean = fg_clipped.mean()
        std  = fg_clipped.std()

        # Z-score the entire volume, then force background to -5
        # so it renders as black and doesn't bias the model.
        normalized_volume = (volume - mean) / (std + 1e-8)
        normalized_volume = np.where(foreground_mask, normalized_volume, -5.0)

    # Final symmetric clip to suppress any remaining outliers.
    return np.clip(normalized_volume, -5.0, 5.0).astype(np.float32)


# --- Legacy functions ---

def min_max_norm(image, lower_q=0.5, upper_q=99.5):
    """Robust min-max normalization using quantiles.
    Use only for evaluating old models (up to version 283).
    Only makes sense for MRI.

    Args:
        image: Input image (tensor).
        lower_q: Lower quantile (default: 0.5).
        upper_q: Upper quantile (default: 99.5).
    
    Returns:
        Normalized image.
    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    
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