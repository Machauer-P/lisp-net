"""
utils/cropping.py
=================
Shared anatomical cropping for dataset converters (HanSeg, FLARE, …).

Strategy
--------
Z-axis : intersection of segmentation extent and image-signal extent.
         Using the intersection avoids including empty slices that exist
         only because the scanner FOV or a registered volume extends
         beyond the actual data (common in MRI registered to CT).
X / Y  : image-signal extent (tight body/head crop) + margin.

All axes receive a configurable safety *margin* in voxels.

Public API
----------
ct_signal_threshold(image_array)  -> float
mri_signal_threshold(image_array) -> float
crop_to_anatomy(image_array, combined_mask, margin, threshold_fn) -> (img, mask)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Threshold strategies
# ---------------------------------------------------------------------------

def ct_signal_threshold(image_array: np.ndarray) -> float:
    """
    CT threshold: -500 HU separates tissue/contrast from air and scanner table.
    Falls back to vmin+ε when the minimum is already above -500 HU
    (e.g. pre-clipped volumes).
    """
    vmin = float(image_array.min())
    if vmin < -500:
        return -500.0
    return vmin + 1e-3


def mri_signal_threshold(image_array: np.ndarray) -> float:
    """
    MRI threshold: 2 % of the maximum signal.

    This cleanly separates background / zero-fill (from registration) from
    tissue. Note: very dark surface voxels (air cavities, scalp edges) may
    sit below the threshold — the crop can therefore be slightly tighter than
    the true anatomy boundary on X / Y. Increase the *margin* if needed.
    """
    return float(image_array.max()) * 0.02


# ---------------------------------------------------------------------------
# Core cropping function
# ---------------------------------------------------------------------------

def crop_to_anatomy(
    image_array: np.ndarray,
    combined_mask: np.ndarray,
    margin: int,
    threshold_fn=None,
) -> tuple:
    """
    Crop a 3-D volume and its label mask to the anatomical region of interest.

    Parameters
    ----------
    image_array   : np.ndarray, shape (Z, Y, X)
    combined_mask : np.ndarray, shape (Z, Y, X)  — integer label map
    margin        : int  — extra voxels added around the detected extent
                    on every axis (before clamping to valid index range).
    threshold_fn  : callable(image_array) -> float, optional
                    Returns the signal threshold; voxels *above* this value
                    are considered foreground.  Defaults to mri_signal_threshold.

    Returns
    -------
    (cropped_image, cropped_mask) — both share the same spatial crop.
    If the extent cannot be determined, the originals are returned unchanged.

    Notes
    -----
    Z behaviour
        The final Z range is the intersection of the segmentation extent
        (expanded by *margin*) and the image-signal extent.  The signal
        clamp prevents padding into empty FOV that exists only because
        the scanner acquired more slices than contain data (or because an MRI
        was registered onto a larger CT grid).

        Consequence: the effective Z margin may be smaller than requested when
        the segmentation extent + margin overshoots the signal boundary.  This
        is intentional; overshooting would include empty / zero-filled slices.

    X / Y behaviour
        The crop is based purely on image-signal extent plus *margin*.
        No further clamping is applied, so the requested margin is always
        honoured as long as the image is large enough.
    """
    if threshold_fn is None:
        threshold_fn = mri_signal_threshold

    threshold = threshold_fn(image_array)
    signal = image_array > threshold

    # Foreground extent along each axis
    z_seg = np.where(np.any(combined_mask > 0, axis=(1, 2)))[0]  # seg Z
    z_img = np.where(np.any(signal,           axis=(1, 2)))[0]   # signal Z
    y_img = np.where(np.any(signal,           axis=(0, 2)))[0]   # signal Y
    x_img = np.where(np.any(signal,           axis=(0, 1)))[0]   # signal X

    # Guard: if any extent is empty, we cannot crop meaningfully
    if len(z_seg) == 0 or len(z_img) == 0 or len(y_img) == 0 or len(x_img) == 0:
        missing = [
            name for name, arr in (
                ("z_seg", z_seg), ("z_img", z_img),
                ("y_img", y_img), ("x_img", x_img),
            )
            if len(arr) == 0
        ]
        print(f"    Warning: empty extent ({', '.join(missing)}) – returning uncropped.")
        return image_array, combined_mask

    # --- Z: seg extent + margin, clamped to signal bounds ---
    z_min = max(0,                       z_seg[0]  - margin)
    z_max = min(image_array.shape[0] - 1, z_seg[-1] + margin)

    # Intersect with signal extent to avoid empty-slice padding
    z_min = max(z_min, z_img[0])
    z_max = min(z_max, z_img[-1])

    if z_min > z_max:
        print("    Warning: segmentation and image-signal extents do not overlap on Z – returning uncropped.")
        return image_array, combined_mask

    # --- Y / X: signal extent + margin ---
    y_min = max(0,                       y_img[0]  - margin)
    y_max = min(image_array.shape[1] - 1, y_img[-1] + margin)

    x_min = max(0,                       x_img[0]  - margin)
    x_max = min(image_array.shape[2] - 1, x_img[-1] + margin)

    cropped_img  = image_array  [z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = combined_mask[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

    return cropped_img, cropped_mask
