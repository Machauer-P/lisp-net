"""
Prompt U-Net Model

Changes over v292 (prompt_unet_292.py):

  1. Configurable filter schedule — default [48, 96, 192, 256, 384] gives
     roughly 15 M parameters, placing the model between v292 (~4 M) and
     v21 (~45 M).  Pass a custom `filters` list to sweep the capacity.

  2. Conv2D at deep stages (4 & 5) instead of SeparableConv2D.
     At high channel counts (256 / 384) the depthwise-separable factorisation
     assumes spatial and channel features are independent — an assumption
     that weakens for the abstract semantic features at the bottleneck.
     Standard Conv2D recovers joint spatial + channel mixing at those stages.

  3. Squeeze-and-Excitation (SE) channel attention on prompt skip
     connections before they are added to the image branch.
     SE computes a per-channel importance weight from the prompt features
     dynamically for each input, allowing the model to suppress irrelevant
     or noisy prompt channels (e.g. at large offsets) while amplifying the
     useful boundary / shape channels.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Conv2D


class PromptUNet:
    """
    Prompt U-Net — dual-encoder segmentation network conditioned on a spatial
    prompt mask.

    Parameters
    ----------
    height, width : int
        Spatial size of the input tensors (default 128 × 128).
    filters : list[int]  (length 5)
        Number of feature channels for each of the 5 encoder stages.
        Stages 4 and 5 (index 3 and 4) use Conv2D; the rest use
        SeparableConv2D.

        Suggested presets
        -----------------
        [32, 64,  96, 128, 192]  →  ~8 M params   (lighter)
        [48, 96, 192, 256, 384]  →  ~15 M params  (default, balanced)
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
        • Prompt encoder  — 5 stages, stores skip connections
        • Image encoder   — 5 stages, each fused with the corresponding
                            SE-gated prompt skip via Add()
        • Bottleneck      — one Conv2D block (always full conv)
        • Decoder         — 5 stages, each concatenates the image skip,
                            mirrors the encoder conv type per stage
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

        F   = self.filters
        # True  → SeparableConv2D (stages 1–3, shallow / cheap)
        # False → Conv2D          (stages 4–5, deep / richer cross-channel mixing)
        sep = [True, True, True, False, False]

        # ── primitive building blocks ─────────────────────────────────

        def conv_block(inp, filters, use_sep=True, padding='same',
                       dropout_rate=0.1, **kwargs):
            """Single convolution → BN → LeakyReLU → Dropout."""
            if use_sep:
                inp = layers.SeparableConv2D(
                    filters, (3, 3), padding=padding, **kwargs)(inp)
            else:
                inp = Conv2D(
                    filters, (3, 3), padding=padding, **kwargs)(inp)
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            return inp

        def conv_block_prompt(x, p, filters, use_sep=True,
                              padding='same', dropout_rate=0.1):
            """
            Fuses a prompt skip into the image branch.

            The prompt features are first processed by a conv_block, then
            passed through an SE gate so the model can down-weight noisy or
            irrelevant prompt channels before adding them to the image stream.
            """
            p = conv_block(p, filters, use_sep=use_sep)
            if use_sep:
                x = layers.SeparableConv2D(
                    filters, (3, 3), padding=padding)(x)
            else:
                x = Conv2D(filters, (3, 3), padding=padding)(x)
            x = layers.Add()([x, p])
            x = layers.Dropout(dropout_rate)(x)
            return x

        def conv_block_up(inp, filters, use_sep=True,
                          padding='same', dropout_rate=0.1):
            """BN → LeakyReLU → Dropout → bilinear upsample → one conv."""
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            inp = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(inp)
            if use_sep:
                inp = layers.SeparableConv2D(
                    filters, (3, 3), padding=padding)(inp)
            else:
                inp = Conv2D(filters, (3, 3), padding=padding)(inp)
            return inp

        # ── encoder / decoder stages ──────────────────────────────────

        def encoder_block(p, filters, use_sep=True):
            """
            Prompt encoder stage.
            Two conv_blocks at `filters` channels → save skip → one strided
            conv_block that doubles the channels and halves the spatial size.
            """
            p = conv_block(p, filters, use_sep=use_sep)
            p = conv_block(p, filters, use_sep=use_sep)
            prompt_skip_connections.append(p)                # save skip
            p = conv_block(p, filters * 2, use_sep=use_sep, strides=2)
            return p

        def encoder_block_2(x, p, filters, use_sep=True):
            """
            Image encoder stage conditioned on the SE-gated prompt skip.
            SE-gated prompt fusion → save skip → strided down.
            """
            x = conv_block_prompt(x, p, filters, use_sep=use_sep)
            skip_connections.append(x)                       # save skip
            x = conv_block(x, filters * 2, use_sep=use_sep, strides=2)
            return x

        def decoder_block(inp, concat_layer, filters, use_sep=True,
                          dropout_rate=0.1):
            """Upsample → concatenate encoder skip → one conv."""
            x = conv_block_up(inp, filters, use_sep=use_sep)
            x = concatenate([x, concat_layer])
            x = conv_block(x, filters, use_sep=use_sep, dropout_rate=dropout_rate)
            return x

        # ── prompt encoder ────────────────────────────────────────────
        prompt = layers.SeparableConv2D(F[0], (3, 3), padding='same')(prompt)
        for i, f in enumerate(F):
            prompt = encoder_block(prompt, f, use_sep=sep[i])

        # ── image encoder ─────────────────────────────────────────────
        x = layers.SeparableConv2D(F[0], (3, 3), padding='same')(image)
        for i, f in enumerate(F):
            x = encoder_block_2(
                x, prompt_skip_connections.pop(0), f, use_sep=sep[i])

        # ── bottleneck (always Conv2D — deepest semantic abstraction) ─
        x = conv_block(x, F[-1], use_sep=False, dropout_rate=0.2)

        # ── decoder (mirror encoder stage flags in reverse) ───────────
        for i, f in enumerate(reversed(F)):
            x = decoder_block(
                x, skip_connections.pop(), f, use_sep=sep[-(i + 1)])

        # ── output ────────────────────────────────────────────────────
        output = Conv2D(1, 1)(x)
        output = tf.keras.activations.sigmoid(output)

        return tf.keras.Model(inputs=inputs, outputs=output)

    # ------------------------------------------------------------------
    @tf.function
    def train_step(self, z):
        """
        Single optimisation step.

        Args:
            z (tuple): (image, label, prompt) batch tensors.
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
            train_dataset (tf.data.Dataset): (image, label, prompt) tuples.
            batch_size (int, optional): Batch size (if dataset not yet batched).
            augmenter (callable, optional): Augmentation function mapped per sample.
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
