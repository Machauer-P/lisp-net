"""
Prompt U-Net v320
=================

Purpose
-------
v320 is a control experiment to answer: *do the performance differences
between v21 and later models (v292–v313) come purely from the new training
data pipeline?*

To isolate that effect, v320 keeps the **v21 architecture and augmentation
exactly** while adopting only the infrastructure improvements of the modern
generation:

  v21  (unchanged)
  ├── Architecture        — Conv2D filter schedule [32, 64, 128, 256, 512],
  │                         bottleneck 1024, Conv2DTranspose decoder
  ├── Augmentation        — identical 10 % probabilities, same geo/photo/prompt
  │                         perturbers (RandomFlip, RandomRotation, …, morph)
  ├── Loss & Optimizer    — binary_crossentropy + ExponentialDecay Adam
  └── Hyperparameters     — 4200 epochs, batch 128, dp 3500, offset 12

  Modernised (different from v21)
  ├── DataGenerator       — current DataGenerator.py (isotropic volumes,
  │                         label-guided 128×128 patch crop, pure-numpy)
  ├── Normalization       — universal_normalization (CT hard-coded, MRI masked
  │                         z-score with percentile clipping)
  ├── Training data       — 3 datasets via DataLoader_npz
  │                         [nako_combined, total_seg_combined, msd_combined]
  ├── train_step()        — decorated with @tf.function (graph mode)
  └── train_epoch()       — native `for z in train_dataset` loop (no manual
                            iter / next)

Rationale: any performance gap between v320 and v21 is attributable solely
to the new training data + preprocessing.  Any gap between v320 and v310–313
is attributable to architecture + augmentation choices.
"""

import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose
from scipy.ndimage import label, binary_erosion, binary_dilation


# ──────────────────────────────────────────────────────────────────────────────
#  v21 augmentation helpers (identical to the original notebook)
# ──────────────────────────────────────────────────────────────────────────────

def _build_geo_aug():
    """Build the v21 geometric augmentation pipeline (applied to x||y||p)."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(128, 128, 4)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05, fill_mode="reflect", interpolation="nearest"),
        layers.RandomZoom((-0.05, 0.05), (-0.05, 0.05),
                         fill_mode="reflect", interpolation="nearest"),
        layers.RandomTranslation((-0.05, 0.05), (-0.05, 0.05),
                                 fill_mode="reflect", interpolation="nearest"),
    ])


def _build_photo_aug():
    """Build the v21 photometric augmentation pipeline (applied to x only)."""
    return tf.keras.Sequential([
        layers.RandomBrightness(factor=0.05, value_range=(0, 1)),
        layers.Lambda(lambda x: x + tf.random.uniform(tf.shape(x), -0.05, 0.05)),
        layers.RandomContrast(factor=0.1),
        layers.GaussianNoise(0.03),
    ])


# ── prompt morphological helpers ─────────────────────────────────────────────

def _cut_out(p_segm, max_fraction_upper=0.25):
    """Simulates user forgetting some pixels."""
    p_eq_1_mask = tf.equal(p_segm, 1)
    num = random.uniform(0.0, max_fraction_upper)
    random_mask = tf.less(tf.random.uniform(shape=tf.shape(p_segm)), num)
    final_mask = tf.logical_and(p_eq_1_mask, random_mask)
    return tf.where(final_mask, tf.constant(0, dtype=p_segm.dtype), p_segm)


def _add_false_positives(p_segm, max_fraction_upper):
    """Randomly adds false positives to the prompt mask."""
    max_fraction = tf.random.uniform([], 0.0, max_fraction_upper)
    background_mask = tf.equal(p_segm, 0)
    random_mask = tf.less(tf.random.uniform(shape=tf.shape(p_segm)), max_fraction)
    add_mask = tf.logical_and(background_mask, random_mask)
    return tf.where(add_mask, tf.constant(1, dtype=p_segm.dtype), p_segm)


def _selective_dilate(mask, kernel_size, min_size):
    """Dilate only connected components larger than min_size (numpy)."""
    mask_2d = np.squeeze(mask)
    labeled, num_features = label(mask_2d)
    dilated_mask = mask_2d.copy()
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if np.sum(component) >= min_size:
            structure = np.ones((kernel_size,) * component.ndim)
            dilated_component = binary_dilation(component, structure=structure)
            dilated_mask[dilated_component] = 1
    if mask.ndim == 3 and mask.shape[2] == 1:
        dilated_mask = dilated_mask[..., np.newaxis]
    return dilated_mask.astype(mask.dtype)


def _tf_selective_dilate(mask, kernel_size, min_size):
    out = tf.numpy_function(_selective_dilate, [mask, kernel_size, min_size], tf.float32)
    out.set_shape(mask.shape)
    return out


def _selective_erode(mask, kernel_size, min_size):
    """Erode only connected components larger than min_size (numpy)."""
    mask_2d = np.squeeze(mask)
    labeled, num_features = label(mask_2d)
    eroded_mask = mask_2d.copy()
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if np.sum(component) >= min_size:
            structure = np.ones((kernel_size,) * component.ndim)
            eroded_component = binary_erosion(component, structure=structure)
            eroded_mask[component] = eroded_component[component]
    if mask.ndim == 3 and mask.shape[2] == 1:
        eroded_mask = eroded_mask[..., np.newaxis]
    return eroded_mask.astype(mask.dtype)


def _tf_selective_erode(mask, kernel_size, min_size):
    eroded = tf.numpy_function(_selective_erode, [mask, kernel_size, min_size], tf.float32)
    eroded.set_shape(mask.shape)
    return eroded


def _random_morphological_perturbation(mask,
                                        max_dilate_kernel, max_erode_kernel,
                                        min_erode_size,   min_dilate_size):
    """Randomly apply erosion (0), dilation (1), or no-op (2)."""
    rand_val = tf.random.uniform([], 0, 3, dtype=tf.int32)

    def erode_fn():
        k = tf.random.uniform([], 1, max_erode_kernel + 1, dtype=tf.int32)
        return _tf_selective_erode(mask, kernel_size=k, min_size=min_erode_size)

    def dilate_fn():
        k = tf.random.uniform([], 1, max_dilate_kernel + 1, dtype=tf.int32)
        return _tf_selective_dilate(mask, kernel_size=k, min_size=min_dilate_size)

    def do_nothing():
        return mask

    return tf.switch_case(rand_val, branch_fns={
        0: erode_fn,
        1: dilate_fn,
        2: do_nothing,
    })


# ──────────────────────────────────────────────────────────────────────────────
#  Model class
# ──────────────────────────────────────────────────────────────────────────────

class PromptUNet:
    """
    Prompt U-Net v320 — exact v21 architecture with modern training infrastructure.

    Parameters
    ----------
    height, width : int
        Spatial size of the input tensors (128 × 128 matching v21).
    """

    def __init__(self, height=128, width=128):
        self.height = height
        self.width  = width

        self.loss       = tf.losses.binary_crossentropy
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        # Build v21 augmenters
        self._geo_aug   = _build_geo_aug()
        self._photo_aug = _build_photo_aug()

        self.this       = self.build()
        self.optimizer  = None  # assigned externally before training

    # ------------------------------------------------------------------
    def build(self) -> tf.keras.Model:
        """
        Builds the exact v21 architecture.

        v21 filter schedule : [32, 64, 128, 256, 512] + 1024 bottleneck
        Decoder             : Conv2DTranspose (not bilinear UpSampling)
        Prompt fusion       : plain Add() (no SE)
        """
        inputs = [
            tf.keras.Input(shape=(self.height, self.width, 1), name='image'),
            tf.keras.Input(shape=(self.height, self.width, 2), name='prompt'),
        ]
        image  = inputs[0]
        prompt = inputs[1]

        prompt_skip_connections = []
        skip_connections        = []

        # ── primitive building blocks ─────────────────────────────────

        def conv_block(inp, filters, padding='same', dropout_rate=0.1, **kwargs):
            """Conv2D → BN → LeakyReLU → Dropout  (v21 convention)."""
            inp = layers.Conv2D(filters, (3, 3), padding=padding, **kwargs)(inp)
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            return inp

        def conv_block_prompt(x, p, filters, padding='same', dropout_rate=0.1):
            """Fuse prompt skip into image branch via Add() — no SE (v21 style)."""
            p = conv_block(p, filters)
            x = layers.Conv2D(filters, (3, 3), padding=padding)(x)
            x = layers.Add()([x, p])
            x = layers.Dropout(dropout_rate)(x)
            return x

        def conv_block_up(inp, filters, padding='same', dropout_rate=0.1, **kwargs):
            """BN → LeakyReLU → Dropout → Conv2DTranspose  (v21 decoder style)."""
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            inp = Conv2DTranspose(filters, (3, 3), padding=padding, **kwargs)(inp)
            return inp

        # ── encoder / decoder stages ──────────────────────────────────

        def encoder_block(p, filters):
            """Prompt encoder stage (v21): 2× conv → save skip → strided down."""
            p = conv_block(p, filters)
            p = conv_block(p, filters)
            prompt_skip_connections.append(p)
            p = conv_block(p, filters * 2, strides=2)
            return p

        def encoder_block_2(x, p, filters):
            """Image encoder stage conditioned on prompt (v21 style)."""
            x = conv_block_prompt(x, p, filters)
            skip_connections.append(x)
            x = conv_block(x, filters * 2, strides=2)
            return x

        def decoder_block(inp, concat_layer, filters, dropout_rate=0.1):
            """Decoder stage: Conv2DTranspose → conv → concatenate → conv (v21)."""
            x = conv_block_up(inp, filters, strides=2)
            x = conv_block(x, filters, dropout_rate=dropout_rate)
            x = concatenate([x, concat_layer])
            x = conv_block(x, filters, dropout_rate=dropout_rate)
            return x

        # ── prompt encoder (v21 filter schedule [32,64,128,256,512]) ─
        prompt = Conv2D(32, (3, 3), padding='same')(prompt)
        prompt = encoder_block(prompt, 32)
        prompt = encoder_block(prompt, 64)
        prompt = encoder_block(prompt, 128)
        prompt = encoder_block(prompt, 256)
        prompt = encoder_block(prompt, 512)

        # ── image encoder ─────────────────────────────────────────────
        x = Conv2D(32, (3, 3), padding='same')(image)
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 32)
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 64)
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 128)
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 256)
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 512)

        # ── bottleneck (1 024 channels, higher dropout) ───────────────
        x = conv_block(x, 1024, dropout_rate=0.2)

        # ── decoder ───────────────────────────────────────────────────
        x = decoder_block(x, skip_connections.pop(), 512)
        x = decoder_block(x, skip_connections.pop(), 256)
        x = decoder_block(x, skip_connections.pop(), 128)
        x = decoder_block(x, skip_connections.pop(), 64)
        x = decoder_block(x, skip_connections.pop(), 32)

        # ── output ────────────────────────────────────────────────────
        output = Conv2D(1, 1)(x)
        output = tf.keras.activations.sigmoid(output)

        return tf.keras.Model(inputs=inputs, outputs=output)

    # ------------------------------------------------------------------
    def v21_augmentation_tf(self, x, y, p, m):
        """
        v21 augmentation — **pure TF ops only** (safe to call directly in
        a `.map()` step without wrapping in `tf.py_function`).

        Stages
        ------
        Photometric  → x only   (10 % probability)
        Geometric    → x, y, p consistently (10 % probability)

        The prompt-morphological step is kept separate in
        `v21_augmentation_morph()` because it requires scipy (numpy) ops.
        """
        # --- photometric (x only) ---
        do_photo = tf.random.uniform([]) < 0.1
        if do_photo:
            x_aug = self._photo_aug(tf.expand_dims(x, 0))   # needs batch dim
            x_aug = tf.squeeze(x_aug, 0)
            x_min = tf.reduce_min(x_aug)
            x_max = tf.reduce_max(x_aug)
            x = (x_aug - x_min) / (x_max - x_min + 1e-8)

        # --- geometric (x, y, p together) ---
        # NOTE: _geo_aug must be called HERE (graph mode), NOT inside
        # tf.py_function, because RandomRotation/RandomZoom have a TF bug
        # with 4-channel tensors when executed inside EagerPyFunc.
        do_geo = tf.random.uniform([]) < 0.1
        if do_geo:
            concatenated = tf.concat([x, y, p], axis=-1)      # (H,W,4)
            concatenated = tf.expand_dims(concatenated, 0)    # (1,H,W,4)
            concatenated = self._geo_aug(concatenated, training=True)
            concatenated = concatenated[0, ...]                # (H,W,4)
            x, y, p = tf.split(concatenated,
                                num_or_size_splits=[1, 1, 2], axis=-1)

        return x, y, p, m

    def v21_augmentation_morph(self, x, y, p):
        """
        v21 prompt-morphological augmentation — scipy/numpy ops.
        Called via `tf.py_function` so it runs eagerly.

        Stage
        -----
        Prompt morph → p (segm channel only, 10 % probability)
        """
        # --- prompt morphological (p segm channel only) ---
        if random.random() < 0.1:
            p_segm = p[..., 1:2]
            p_segm = _cut_out(p_segm, max_fraction_upper=0.20)
            p_segm = _add_false_positives(p_segm, max_fraction_upper=0.001)
            p_segm = tf.squeeze(p_segm)
            p_segm = _random_morphological_perturbation(
                p_segm,
                max_dilate_kernel=2, max_erode_kernel=2,
                min_erode_size=30,  min_dilate_size=30,
            )
            p_segm = tf.expand_dims(p_segm, -1)
            p = tf.concat([p[..., 0:1], p_segm], axis=-1)

        return x, y, p   # no batch dim; .batch() adds it later

    def v21_augmentation(self, x, y, p):
        """
        Legacy single-call wrapper kept for compatibility.
        Prefer calling v21_augmentation_tf + v21_augmentation_morph
        in separate .map() steps (see notebook pipeline).

        WARNING: calling this inside tf.py_function will crash because
        _geo_aug(RandomRotation) cannot handle 4-ch tensors in EagerPyFunc.
        """
        x, y, p, _ = self.v21_augmentation_tf(x, y, p, None)
        x, y, p    = self.v21_augmentation_morph(x, y, p)
        return x, y, p

    # ------------------------------------------------------------------
    @tf.function
    def train_step(self, z):
        """
        Single optimisation step — decorated with @tf.function (graph mode).
        z = (image, label, prompt, modality) batch.
        """
        with tf.GradientTape() as tape:
            y_pred = self.this([z[0], z[2]], training=True)
            loss   = self.loss(z[1], y_pred)
        grads = tape.gradient(loss, self.this.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.this.trainable_variables))
        self.train_loss.update_state(loss)

    def train_epoch(self, train_dataset):
        """
        One training epoch — native iteration (for z in train_dataset).
        The dataset is expected to be already shuffled, augmented, batched,
        and prefetched by the caller (persistent pipeline pattern).
        """
        for z in train_dataset:
            self.train_step(z)
