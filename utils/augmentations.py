import tensorflow as tf
import numpy as np
import math
from scipy.ndimage import label, binary_erosion, binary_dilation
from tensorflow.keras import Sequential, layers


def synced_geometric_aug(images, masks):
    """
    Applies identical random flip, rotation, zoom, and translation.
    images: Continuous tensor [H, W, C_img] -> uses BILINEAR
    masks: Categorical tensor [H, W, C_mask] -> uses NEAREST
    """
    # --- 1. Synchronized Flip ---
    # Create a unified seed for this specific sample
    seed = tf.random.uniform([2], 0, 2**31-1, dtype=tf.int32)
    images = tf.image.stateless_random_flip_left_right(images, seed=seed)
    masks = tf.image.stateless_random_flip_left_right(masks, seed=seed)

    # --- 2. Generate Random Affine Parameters (Matches Keras bounds) ---
    theta = tf.random.uniform([], -0.05 * 2 * math.pi, 0.05 * 2 * math.pi) # Rotation
    zx = tf.random.uniform([], 0.95, 1.05)                                 # Zoom X
    zy = tf.random.uniform([], 0.95, 1.05)                                 # Zoom Y
    
    h = tf.cast(tf.shape(images)[0], tf.float32)
    w = tf.cast(tf.shape(images)[1], tf.float32)
    tx = tf.random.uniform([], -0.05, 0.05) * w                            # Translate X
    ty = tf.random.uniform([], -0.05, 0.05) * h                            # Translate Y
    
    cx, cy = w / 2.0, h / 2.0 # Center coordinates

    # --- 3. Compute Inverse Affine Matrix Elements ---
    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)

    a0 = cos_t / zx
    a1 = sin_t / zx
    a2 = -cx * a0 - cy * a1 + cx - tx

    b0 = -sin_t / zy
    b1 = cos_t / zy
    b2 = -cx * b0 - cy * b1 + cy - ty

    # TF expects an 8-element vector [a0, a1, a2, b0, b1, b2, c0, c1]
    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    transforms = tf.expand_dims(transforms, 0) # Add batch dim for the op

    # --- 4. Apply Transforms with Explicit Interpolations ---
    # ImageProjectiveTransformV3 requires a 4D tensor (Batch, H, W, C)
    images_exp = tf.expand_dims(images, 0)
    masks_exp = tf.expand_dims(masks, 0)

    # The raw C++ op requires a fill_value tensor, even if REFLECT ignores it
    dummy_fill = tf.constant(0.0, dtype=tf.float32)

    images_aug = tf.raw_ops.ImageProjectiveTransformV3(
        images=images_exp,
        transforms=transforms,
        output_shape=tf.shape(images)[:2],
        fill_value=dummy_fill,          # <-- Added required argument
        interpolation="BILINEAR", 
        fill_mode="REFLECT"
    )
    
    masks_aug = tf.raw_ops.ImageProjectiveTransformV3(
        images=masks_exp,
        transforms=transforms,
        output_shape=tf.shape(masks)[:2],
        fill_value=dummy_fill,          # <-- Added required argument
        interpolation="NEAREST", 
        fill_mode="REFLECT"
    )

    return tf.squeeze(images_aug, 0), tf.squeeze(masks_aug, 0)


def random_gaussian_noise(x, min_std=0.0, max_std=0.10):
    std = tf.random.uniform([], min_std, max_std)
    noise = tf.random.normal(shape=tf.shape(x), stddev=std)
    return x + noise


def random_gamma(x, min_gamma=0.7, max_gamma=1.5):
    """Applies a random gamma curve to non-linearly shift contrast."""
    gamma = tf.random.uniform([], min_gamma, max_gamma)
    
    # Safely map [-5.0, 5.0] Z-score bounds to [0, 1] before applying gamma power curve
    x_norm = (x + 5.0) / 10.0
    x_norm = tf.clip_by_value(x_norm, 0.0, 1.0)
    
    x_gamma = tf.pow(x_norm, gamma)
    
    # Map back to original [-5.0, 5.0] distribution
    return (x_gamma * 10.0) - 5.0


def capped_dropout_blockwise(mask,
                  max_strokes=3,
                  max_stroke_len=20,
                  max_stroke_width=3,
                  min_component_size=30,
                  max_remove_fraction=0.2):
    """
    Scribble dropout with component awareness AND damage cap.
    """

    mask_np = np.squeeze(mask)
    h, w = mask_np.shape
    result = mask_np.copy()

    labeled, num_features = label(mask_np)

    for comp_id in range(1, num_features + 1):
        component = (labeled == comp_id)
        comp_size = np.sum(component)

        # Skip small components
        if comp_size < min_component_size:
            continue

        ys, xs = np.where(component)

        # Random scribble params
        num_strokes = np.random.randint(0, max_strokes)
        stroke_length = np.random.randint(5, max_stroke_len)
        stroke_width = np.random.randint(1, max_stroke_width)

        removed_pixels = 0
        max_remove_pixels = int(comp_size * max_remove_fraction)

        for _ in range(num_strokes):
            if removed_pixels >= max_remove_pixels:
                break

            if len(xs) == 0:
                break

            idx = np.random.randint(0, len(xs))
            y, x = ys[idx], xs[idx]

            for _ in range(stroke_length):
                dy = np.random.randint(-1, 2)
                dx = np.random.randint(-1, 2)

                y = np.clip(y + dy, 0, h - 1)
                x = np.clip(x + dx, 0, w - 1)

                for wy in range(-stroke_width, stroke_width + 1):
                    for wx in range(-stroke_width, stroke_width + 1):
                        yy = np.clip(y + wy, 0, h - 1)
                        xx = np.clip(x + wx, 0, w - 1)

                        if component[yy, xx] == 1 and result[yy, xx] == 1:
                            result[yy, xx] = 0
                            removed_pixels += 1

                            if removed_pixels >= max_remove_pixels:
                                break
                    if removed_pixels >= max_remove_pixels:
                        break
    
    result = np.expand_dims(result, -1)
    return result.astype(np.float32)


def false_positives_blockwise(p_segm, max_block_size_ratio=0.05, max_blocks=3):
    p_segm = tf.cast(p_segm, tf.float32)

    h = tf.shape(p_segm)[0]
    w = tf.shape(p_segm)[1]

    p_segm_2d = tf.squeeze(p_segm, axis=-1)

    # --- max size ---
    max_block_h = tf.maximum(1, tf.cast(tf.cast(h, tf.float32) * max_block_size_ratio, tf.int32))
    max_block_w = tf.maximum(1, tf.cast(tf.cast(w, tf.float32) * max_block_size_ratio, tf.int32))

    num_blocks = tf.random.uniform([], 1, max_blocks + 1, dtype=tf.int32)

    # radii
    radii_y = tf.random.uniform([num_blocks], 1, max_block_h // 2 + 1, dtype=tf.int32)
    radii_x = tf.random.uniform([num_blocks], 1, max_block_w // 2 + 1, dtype=tf.int32)

    ys = tf.random.uniform([num_blocks], 0, h, dtype=tf.int32)
    xs = tf.random.uniform([num_blocks], 0, w, dtype=tf.int32)

    # grid
    y_grid = tf.range(h, dtype=tf.float32)[:, tf.newaxis]
    x_grid = tf.range(w, dtype=tf.float32)[tf.newaxis, :]

    def draw_blob(i):
        cy = tf.cast(ys[i], tf.float32)
        cx = tf.cast(xs[i], tf.float32)
        ry = tf.cast(radii_y[i], tf.float32)
        rx = tf.cast(radii_x[i], tf.float32)

        # --- random rotation ---
        theta = tf.random.uniform([], 0, 2 * 3.14159)

        # shift to center
        x = x_grid - cx
        y = y_grid - cy

        # rotate coordinates
        xr = x * tf.cos(theta) + y * tf.sin(theta)
        yr = -x * tf.sin(theta) + y * tf.cos(theta)

        # normalized radial distance (Gaussian-style)
        dist = tf.square(xr / rx) + tf.square(yr / ry)

        # --- smooth blob (less star-like) ---
        blob = tf.exp(-dist * 2.5)  # control sharpness

        # random threshold per blob
        thresh = tf.random.uniform([], 0.3, 0.7)

        return blob > thresh

    blob_masks = tf.map_fn(
        draw_blob,
        tf.range(num_blocks),
        fn_output_signature=tf.bool
    )

    blob_mask = tf.reduce_any(blob_masks, axis=0)

    # apply only on background
    background_mask = tf.equal(p_segm_2d, 0)
    add_mask = tf.logical_and(background_mask, blob_mask)

    out = tf.where(add_mask, tf.ones_like(p_segm_2d), p_segm_2d)

    return out[..., tf.newaxis]


def selective_dilate(mask, kernel_size, min_size):
    """
    Dilate only connected components larger than a relative size threshold.

    min_size is now interpreted as a FRACTION of the total image area (e.g. 0.01 = 1%).
    """

    mask_2d = np.squeeze(mask)
    h, w = mask_2d.shape
    image_size = h * w

    labeled, num_features = label(mask_2d)
    dilated_mask = mask_2d.copy()

    for i in range(1, num_features + 1):
        component = (labeled == i)
        component_size = np.sum(component)

        # relative size instead of absolute pixel count
        relative_size = component_size / image_size

        if relative_size >= min_size:
            structure = np.ones((kernel_size,) * component.ndim)
            dilated_component = binary_dilation(component, structure=structure)
            dilated_mask[dilated_component] = 1

    if mask.ndim == 3 and mask.shape[2] == 1:
        dilated_mask = dilated_mask[..., np.newaxis]

    return dilated_mask.astype(mask.dtype)


def tf_selective_dilate(mask, kernel_size, min_size):
    mask_shape = mask.shape
    out = tf.numpy_function(selective_dilate, [mask, kernel_size, min_size], tf.float32)
    out.set_shape(mask.shape)
    return out


def selective_erode(mask, kernel_size, min_size):
    """
    Erode only connected components larger than a relative size threshold.

    min_size is interpreted as a FRACTION of the total image area (e.g. 0.01 = 1%).
    """

    mask_2d = np.squeeze(mask)
    h, w = mask_2d.shape
    image_size = h * w

    labeled, num_features = label(mask_2d)
    eroded_mask = mask_2d.copy()

    for i in range(1, num_features + 1):
        component = (labeled == i)
        component_size = np.sum(component)

        # relative size instead of absolute pixel count
        relative_size = component_size / image_size

        if relative_size >= min_size:
            structure = np.ones((kernel_size,) * component.ndim)
            eroded_component = binary_erosion(component, structure=structure)
            eroded_mask[component] = eroded_component[component]

    if mask.ndim == 3 and mask.shape[2] == 1:
        eroded_mask = eroded_mask[..., np.newaxis]

    return eroded_mask.astype(mask.dtype)


def tf_selective_erode(mask, kernel_size, min_size):
    mask_shape = mask.shape
    eroded = tf.numpy_function(selective_erode, [mask, kernel_size, min_size], tf.float32)
    eroded.set_shape(mask_shape)
    return eroded


def random_morphological_perturbation(mask, max_dilate_kernel, max_erode_kernel, min_erode_size, min_dilate_size):
    """
    Randomly apply Erosion or Dilation.
    """
    rand_val = tf.random.uniform([], 0, 2, dtype=tf.int32)
   
    def erode_fn():
        k = tf.random.uniform([], 2, max_erode_kernel + 1, dtype=tf.int32)
        return tf_selective_erode(mask, kernel_size=k, min_size=min_erode_size)
    
    def dilate_fn():
        k = tf.random.uniform([], 2, max_dilate_kernel + 1, dtype=tf.int32)
        return tf_selective_dilate(mask, kernel_size=k, min_size=min_dilate_size)
    
    return tf.switch_case(rand_val, branch_fns={
        0: erode_fn,
        1: dilate_fn
    })


class PromptUNetAugmenter:
    """
    Augmentation Pipeline for PromptUNet.
    Callable to elegantly drop into tf.data.Dataset.map(...)
    """

    def __init__(self,
                 prob_photo=0.45,
                 prob_gamma=0.45,
                 prob_noise=0.40,
                 prob_independent_noise=0.5,
                 prob_geometric=0.85,
                 prob_morph=0.6,
                 prob_dropout=0.4,
                 prob_false_pos=0.6,
                 gamma_range=(0.7, 1.5),
                 noise_std_range=(0.0, 0.10),
                 independent_noise_std_range=(0.0, 0.010),
                 morph_params=None,
                 false_pos_params=None):
        """
        Configuration for the data augmentation steps. 
        Defaults are set exactly to the baseline script values.
        """
        self.prob_photo = prob_photo
        self.prob_gamma = prob_gamma
        self.prob_noise = prob_noise
        self.prob_independent_noise = prob_independent_noise
        self.prob_geometric = prob_geometric
        self.prob_morph = prob_morph
        self.prob_dropout = prob_dropout
        self.prob_false_pos = prob_false_pos

        self.gamma_range = gamma_range
        self.noise_std_range = noise_std_range
        self.independent_noise_std_range = independent_noise_std_range

        self.morph_params = morph_params or {
            "max_dilate_kernel": 2,
            "max_erode_kernel": 2,
            "min_erode_size": 0.0015,
            "min_dilate_size": 0.0015
        }

        self.false_pos_params = false_pos_params or {
            "max_block_size_ratio": 0.1,
            "max_blocks": 3
        }

        # Keras Sequential Model for Photo Augmentation
        self.photo_aug = Sequential([
            layers.RandomBrightness(factor=0.1, value_range=(-5.0, 5.0)),
            layers.RandomContrast(factor=0.15)
        ])

    def __call__(self, x, y, p, m=None):
        """
        Main augmentation logic to map over the dataset. 
        Input:
            x: Input CT/MRI
            y: Ground Truth Label
            p: Prompt
            m: Optional Modality flag (1.0 for MRI, 0.0 for CT)
        """
        p_x, p_y = tf.split(p, num_or_size_splits=2, axis=-1)
        
        # Determine modality upfront
        if m is not None:
            is_mri = tf.reduce_mean(tf.cast(m, tf.float32)) > 0.5
        else:
            is_mri = tf.constant(True, dtype=tf.bool)
        
        # --- x, p_x (Photometric) ---
        xp = tf.concat([x, p_x], axis=-1)  
        xp = tf.cast(xp, tf.float32)
        
        def apply_photo_mri(xp_in):
            if tf.random.uniform([]) < self.prob_photo:
                xp_in = self.photo_aug(xp_in, training=True) 
                xp_in = tf.cast(xp_in, tf.float32)
            if tf.random.uniform([]) < self.prob_gamma:
                xp_in = random_gamma(xp_in, min_gamma=self.gamma_range[0], max_gamma=self.gamma_range[1])
                xp_in = tf.cast(xp_in, tf.float32)
            return xp_in
            
        def apply_photo_ct(xp_in):
            # No Gamma for CT to preserve physical HU correlations
            if tf.random.uniform([]) < self.prob_photo:
                xp_in = self.photo_aug(xp_in, training=True) 
                xp_in = tf.cast(xp_in, tf.float32)
            return xp_in

        xp = tf.cond(is_mri, lambda: apply_photo_mri(xp), lambda: apply_photo_ct(xp))
            
        x, p_x = tf.split(xp, num_or_size_splits=2, axis=-1)

        if len(x.shape) > len(y.shape):
            x = x[..., :1]
            
        # --- x, y, p (Joint Geometric)---
        continuous_imgs = tf.concat([x, p_x], axis=-1) 
        categorical_masks = tf.concat([y, p_y], axis=-1)

        # Apply synchronized transformations with correct interpolations
        if tf.random.uniform([]) < self.prob_geometric:
            continuous_imgs, categorical_masks = synced_geometric_aug(continuous_imgs, categorical_masks)

        # Disassemble back into original variables
        x, p_x = tf.split(continuous_imgs, num_or_size_splits=2, axis=-1)
        y, p_y = tf.split(categorical_masks, num_or_size_splits=2, axis=-1)

        # --- Only p_y (Artifacts & Morphologies) ---
        if tf.random.uniform([]) < self.prob_morph: 
            p_y = random_morphological_perturbation(
                p_y, 
                max_dilate_kernel=self.morph_params["max_dilate_kernel"],
                max_erode_kernel=self.morph_params["max_erode_kernel"],
                min_erode_size=self.morph_params["min_erode_size"],
                min_dilate_size=self.morph_params["min_dilate_size"]
            )
            
        if tf.random.uniform([]) < self.prob_dropout:
            # Forgetting
            p_y = tf.numpy_function(func=capped_dropout_blockwise, inp=[p_y], Tout=tf.float32) 
            p_y.set_shape([128, 128, 1])
            
        if tf.random.uniform([]) < self.prob_false_pos:
            # Learning to ignore random artefacts
            p_y = false_positives_blockwise(
                p_y, 
                max_block_size_ratio=self.false_pos_params["max_block_size_ratio"], 
                max_blocks=self.false_pos_params["max_blocks"]
            )
            
        p_y = tf.squeeze(p_y)
        p_y = tf.expand_dims(p_y, -1)
        p = tf.concat([p_x, p_y], axis=-1)
        
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        p = tf.cast(p, tf.float32)
        
        # --- Late / Context-Aware Noise (Added after interpolations to prevent blurring) ---
        def apply_noise_mri(x_in, p_in):
            # Joint noise
            if tf.random.uniform([]) < self.prob_noise:
                xp_in = tf.concat([x_in, p_in[..., 0:1]], axis=-1)
                xp_in = random_gaussian_noise(xp_in, min_std=0.0, max_std=0.10) 
                x_in, p_x_after = tf.split(xp_in, 2, axis=-1)
                p_in = tf.concat([p_x_after, p_in[..., 1:2]], axis=-1)
                 
            # Independent noise on image only
            if tf.random.uniform([]) < self.prob_independent_noise:
                x_in = random_gaussian_noise(x_in, min_std=0.0, max_std=0.010)
                 
            return x_in, p_in

        def apply_noise_ct(x_in, p_in):
            # Turned off completely for CT to preserve explicit HU calibration
            return x_in, p_in

        # Apply functionally based on evaluated modality flag
        x, p = tf.cond(is_mri, lambda: apply_noise_mri(x, p), lambda: apply_noise_ct(x, p))
            
        return x, y, p
