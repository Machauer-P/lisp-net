"""
Prompt U-Net v310
=================

Changes over v300 (prompt_unet_300.py):

  1. No SE Attention — Squeeze-and-Excitation channel gates removed from the
     prompt skip fusion.  The model now simply adds the prompt skip directly
     to the image branch, identical to the simpler additive fusion used before
     v300.  Motivation: ablate whether SE attention actually contributes on
     the new (scale-augmented) training distribution.

  2. Pure Conv2D — SeparableConv2D removed at ALL stages (was only in stages
     1–3 in v300).  Every convolution is now a standard 2D convolution.
     Rationale: with scale augmentation the shallow stages now see both
     fine-grained (128×128 crop) and coarser (256→128 down-sampled) textures
     simultaneously.  Separable convolutions assume spatial / channel
     independence that may not hold across such a wide scale range.

  3. Mixed precision (float16) — same as v300.  LossScaleOptimizer wraps
     Adam; gradients are computed in float32 via automatic loss scaling.

Filter schedule and all other hyper-parameters are unchanged from v300:
  Default: [48, 96, 192, 256, 384]  (~15 M trainable params)

Training data changes (DataGenerator.py):
  - Scale Augmentation: 50% chance of 128×128 literal crop (preserves
    1×1 mm physical pixel spacing); 50% chance of a random quadratic
    crop in [128, 256] pixels that is then bilinearly resized to 128×128,
    teaching scale invariance without exceeding the recommended 2× downsample
    ratio (keeps small structures ≥ ~5 px after resize).
  - Leakage fix: crop origin is now computed exclusively from the Support /
    Prompt label (`total_label_r`), not the unknown target label.  In the
    interactive-segmentation use-case the user always provides the prompt,
    so this is legal and clinically realistic.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Conv2D


class PromptUNet:
    """
    Prompt U-Net v310 — pure Conv2D, no SE, float32.

    Parameters
    ----------
    height, width : int
        Spatial size of the input tensors (default 128 × 128).
    filters : list[int]  (length 5)
        Number of feature channels for each of the 5 encoder stages.

        Suggested presets
        -----------------
        [32, 64,  96, 128, 192]  →  ~5 M params   (lighter)
        [48, 96, 192, 256, 384]  →  ~15 M params  (default, matches v300)
        [64, 128, 256, 512, 512] →  ~45 M params  (matches v21)
    """

    def __init__(self, height=128, width=128, filters=None):
        self.height  = height
        self.width   = width
        self.filters = filters if filters is not None else [48, 96, 192, 256, 384]
        assert len(self.filters) == 5, "Provide exactly 5 stage filter counts."

        self.loss       = tf.losses.binary_crossentropy
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.this      = self.build()
        self.optimizer = None  # Assigned externally before training

    # ------------------------------------------------------------------
    def build(self) -> tf.keras.Model:
        """
        Builds and returns the Keras functional model.

        Architecture overview
        ---------------------
        • Prompt encoder  — 5 stages, pure Conv2D, stores skip connections
        • Image encoder   — 5 stages, pure Conv2D, each fused with the
                            corresponding prompt skip via Add()  (no SE gate)
        • Bottleneck      — one Conv2D block
        • Decoder         — 5 stages (pure Conv2D), concatenates image skip
        • Output          — 1×1 Conv + sigmoid
        """
        inputs = [
            tf.keras.Input(shape=(self.height, self.width, 1), name='image'),
            tf.keras.Input(shape=(self.height, self.width, 2), name='prompt'),
        ]
        image  = inputs[0]
        prompt = inputs[1]

        prompt_skip_connections = []
        skip_connections        = []

        F = self.filters

        # ── primitive building blocks ─────────────────────────────────

        def conv_block(inp, filters, padding='same', dropout_rate=0.1, **kwargs):
            """Single Conv2D → BN → LeakyReLU → Dropout."""
            inp = Conv2D(filters, (3, 3), padding=padding, **kwargs)(inp)
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            return inp

        def conv_block_prompt(x, p, filters, padding='same', dropout_rate=0.1):
            """
            Fuses a prompt skip into the image branch (simple additive fusion,
            no SE gate — SE ablated in v310).
            """
            p = conv_block(p, filters)
            x = Conv2D(filters, (3, 3), padding=padding)(x)
            x = layers.Add()([x, p])
            x = layers.Dropout(dropout_rate)(x)
            return x

        def conv_block_up(inp, filters, padding='same', dropout_rate=0.1):
            """BN → LeakyReLU → Dropout → bilinear upsample → Conv2D."""
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            inp = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(inp)
            inp = Conv2D(filters, (3, 3), padding=padding)(inp)
            return inp

        # ── encoder / decoder stages ──────────────────────────────────

        def encoder_block(p, filters):
            """
            Prompt encoder stage.
            Two conv_blocks at `filters` channels → save skip → one strided
            conv_block that doubles the channels and halves the spatial size.
            """
            p = conv_block(p, filters)
            p = conv_block(p, filters)
            prompt_skip_connections.append(p)        # save skip
            p = conv_block(p, filters * 2, strides=2)
            return p

        def encoder_block_2(x, p, filters):
            """
            Image encoder stage conditioned on the prompt skip.
            Prompt fusion (Add) → save skip → strided down.
            """
            x = conv_block_prompt(x, p, filters)
            skip_connections.append(x)               # save skip
            x = conv_block(x, filters * 2, strides=2)
            return x

        def decoder_block(inp, concat_layer, filters, dropout_rate=0.1):
            """Upsample → concatenate encoder skip → one conv."""
            x = conv_block_up(inp, filters)
            x = concatenate([x, concat_layer])
            x = conv_block(x, filters, dropout_rate=dropout_rate)
            return x

        # ── prompt encoder ────────────────────────────────────────────
        prompt = Conv2D(F[0], (3, 3), padding='same')(prompt)
        for f in F:
            prompt = encoder_block(prompt, f)

        # ── image encoder ─────────────────────────────────────────────
        x = Conv2D(F[0], (3, 3), padding='same')(image)
        for f in F:
            x = encoder_block_2(x, prompt_skip_connections.pop(0), f)

        # ── bottleneck ────────────────────────────────────────────────
        x = conv_block(x, F[-1], dropout_rate=0.2)

        # ── decoder (mirror encoder in reverse) ───────────────────────
        for f in reversed(F):
            x = decoder_block(x, skip_connections.pop(), f)

        # ── output ────────────────────────────────────────────────────
        output = Conv2D(1, 1)(x)
        output = tf.keras.activations.sigmoid(output)

        return tf.keras.Model(inputs=inputs, outputs=output)

    # ------------------------------------------------------------------
    @tf.function
    def train_step(self, z):
        """
        Single optimisation step with mixed-precision loss scaling.

        Args:
            z (tuple): (image, label, prompt, modality) batch tensors.
        """
        with tf.GradientTape() as tape:
            y_pred      = self.this([z[0], z[2]], training=True)
            loss        = self.loss(z[1], y_pred)
            scaled_loss = self.optimizer.scale_loss(loss)

        scaled_grads = tape.gradient(scaled_loss, self.this.trainable_variables)
        self.optimizer.apply_gradients(
            zip(scaled_grads, self.this.trainable_variables))
        self.train_loss.update_state(loss)

    def train_epoch(self, train_dataset, batch_size=None, augmenter=None):
        """
        Runs one training epoch over the provided dataset.

        Args:
            train_dataset (tf.data.Dataset): (image, label, prompt, modality).
            batch_size (int, optional): Batch size (if dataset not yet batched).
            augmenter (callable, optional): Augmentation function per sample.
        """
        if augmenter is not None or batch_size is not None:
            train_dataset = train_dataset.shuffle(256)
            if augmenter is not None:
                train_dataset = train_dataset.map(
                    augmenter, num_parallel_calls=tf.data.AUTOTUNE)
            if batch_size is not None:
                train_dataset = train_dataset.batch(
                    batch_size, drop_remainder=True)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        for z in train_dataset:
            self.train_step(z)
