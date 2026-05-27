"""
Prompt U-Net v313
=================

Changes over v312:

  1. SE Attention re-enabled — Squeeze-and-Excitation channel gates are back on
     every prompt skip connection, identical to v311.  This gives v313 the full
     combination of:
       • Pure Conv2D everywhere          (from v310 onward)
       • SE channel-attention on prompts (from v311)
       • Float32 training, plain Adam    (from v312)

  Ablation matrix so far:
    v310 — Conv2D, no SE,  float16 (mixed precision + LossScaleOptimizer)
    v311 — Conv2D, SE,     float16
    v312 — Conv2D, no SE,  float32
    v313 — Conv2D, SE,     float32   ← this file

  v312 vs v313 isolates: does SE attention help under float32 training?
  v311 vs v313 isolates: does float32 (vs float16) matter when SE is present?

Filter schedule and all other hyper-parameters are unchanged from v310–312:
  Default: [48, 96, 192, 256, 384]  (~15 M trainable params)
  Scale augmentation + leakage fix (crop origin from support label only).
  Mixed precision: DISABLED — pure float32 throughout.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Conv2D


class PromptUNet:
    """
    Prompt U-Net v313 — pure Conv2D, SE Attention enabled, float32 training.

    Parameters
    ----------
    height, width : int
        Spatial size of the input tensors (default 128 × 128).
    filters : list[int]  (length 5)
        Number of feature channels for each of the 5 encoder stages.

        Suggested presets
        -----------------
        [32, 64,  96, 128, 192]  →  ~5 M params   (lighter)
        [48, 96, 192, 256, 384]  →  ~15 M params  (default, matches v310–312)
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
                            corresponding SE-gated prompt skip via Add()
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

        def se_block(x, filters, ratio=4):
            """
            Squeeze-and-Excitation channel attention.

            Globally average-pools the spatial dimensions to summarise each
            channel as a scalar (Squeeze), then passes through two Dense
            layers to produce a sigmoid gate in [0, 1] per channel (Excite).
            The input is multiplied element-wise by that gate, so channels
            judged unimportant for the current input are suppressed.
            """
            s = layers.GlobalAveragePooling2D(keepdims=True)(x)      # (1, 1, C)
            s = layers.Dense(max(filters // ratio, 1), activation='relu')(s)
            s = layers.Dense(filters, activation='sigmoid')(s)        # (1, 1, C)
            return layers.Multiply()([x, s])

        def conv_block_prompt(x, p, filters, padding='same', dropout_rate=0.1):
            """
            Fuses a prompt skip into the image branch via SE gate and Add().
            """
            p = conv_block(p, filters)
            p = se_block(p, filters)                         # ← SE gate
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
            Image encoder stage conditioned on the SE-gated prompt skip.
            Prompt fusion (SE + Add) → save skip → strided down.
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
        Single optimisation step (float32 — no loss scaling).

        Args:
            z (tuple): (image, label, prompt, modality) batch tensors.
        """
        with tf.GradientTape() as tape:
            y_pred = self.this([z[0], z[2]], training=True)
            loss   = self.loss(z[1], y_pred)

        grads = tape.gradient(loss, self.this.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.this.trainable_variables))
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
