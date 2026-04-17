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


def universeg_normalization(volume, modality="CT"):
    """
    Intensity normalization matching UniverSeg's training convention.
    Produces float32 values in [0, 1].

    Applied to the raw 3-D volume so that cross-slice relative brightness is
    preserved — identical to how UniverSeg was trained.

    CT  : clip to [-500, 1000] HU (soft-tissue window used by UniverSeg)
           → min-max to [0, 1]:  (HU + 500) / 1500
    MRI : clip to 0.5–99.5 percentile of the full volume
           → min-max to [0, 1]

    Args:
        volume   : np.ndarray of any shape — RAW intensities (before any
                   z-score or other normalization is applied).
        modality : str — "CT" or anything else for MRI/other.

    Returns:
        np.ndarray (float32), values clipped to [0, 1].
    """
    volume = np.asarray(volume, dtype=np.float32)

    if modality == "CT":
        v_min, v_max = -500.0, 1000.0
        volume = np.clip(volume, v_min, v_max)
    else:  # MRI / Other
        v_min = float(np.percentile(volume, 0.5))
        v_max = float(np.percentile(volume, 99.5))
        volume = np.clip(volume, v_min, v_max)

    return ((volume - v_min) / (v_max - v_min + 1e-8)).astype(np.float32)


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


def shaping(tensor, h=128, w=128, binary=False):
    """Ensure a tensor has shape (1, H, W, C) for model inference.

    Handles the following input shapes:
        (H, W, C)  →  (1, H, W, C)
        (1, H, W)  →  (1, H, W, 1)
        (H, W)     →  (1, H, W, 1)

    Args:
        tensor : tf.Tensor or numpy array.
        h, w   : Target spatial dimensions (default 128×128).
        binary : Use nearest-neighbor resizing for binary masks.

    Returns:
        tf.Tensor with shape (1, H, W, 1) or (1, H, W, 2).

    Raises:
        ValueError if the shape cannot be resolved to a valid inference shape.
    """
    import tensorflow as tf

    tensor = tf.cast(tensor, tf.float32)

    if len(tensor.shape) == 2:
        # (H, W) → (1, H, W, 1)
        tensor = tensor[tf.newaxis, ..., tf.newaxis]

    elif len(tensor.shape) == 3:
        s = tensor.shape
        if s[0] > 1 and s[1] > 1 and s[2] in (1, 2):
            # (H, W, C) → (1, H, W, C)
            tensor = tensor[tf.newaxis, ...]
        elif s[0] == 1 and s[1] > 1 and s[2] > 1:
            # (1, H, W) → (1, H, W, 1)
            tensor = tensor[..., tf.newaxis]

    # Resize spatial dims if needed
    if tensor.shape[1] != h or tensor.shape[2] != w:
        method = 'nearest' if binary else 'bilinear'
        tensor = tf.image.resize(tensor, [h, w], method=method)

    # Validate final shape
    if not (tensor.shape[0] == 1
            and tensor.shape[1] == h
            and tensor.shape[2] == w
            and tensor.shape[3] in (1, 2)):
        raise ValueError(f'shaping() failed — unexpected result shape: {tensor.shape}')

    return tensor