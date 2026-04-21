"""
Prompt U-Net v314
=================

Changes over v313:
  1. Only depthwise separable convolutions (SeparableConv2D) instead of standard Conv2D
     for all 3x3 spatial convolutions.

Filter schedule and all other hyper-parameters are unchanged from v313:
  Default: [48, 96, 192, 256, 384]  (~15 M trainable params)
  SeparableConv2D everywhere (except final 1x1 output).
  SE Attention re-enabled on prompt skip connections.
  Pure float32 training.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Conv2D, SeparableConv2D

class PromptUNet:
    """
    Prompt U-Net v314 — SeparableConv2D everywhere, SE Attention enabled, float32 training.

    Parameters
    ----------
    height, width : int
        Spatial size of the input tensors (default 128 × 128).
    filters : list[int]  (length 5)
        Number of feature channels for each of the 5 encoder stages.
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
            """Single SeparableConv2D → BN → LeakyReLU → Dropout."""
            inp = SeparableConv2D(filters, (3, 3), padding=padding, **kwargs)(inp)
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            return inp

        def se_block(x, filters, ratio=4):
            """Squeeze-and-Excitation channel attention."""
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
            x = SeparableConv2D(filters, (3, 3), padding=padding)(x)
            x = layers.Add()([x, p])
            x = layers.Dropout(dropout_rate)(x)
            return x

        def conv_block_up(inp, filters, padding='same', dropout_rate=0.1):
            """BN → LeakyReLU → Dropout → bilinear upsample → SeparableConv2D."""
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            inp = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(inp)
            inp = SeparableConv2D(filters, (3, 3), padding=padding)(inp)
            return inp

        # ── encoder / decoder stages ──────────────────────────────────

        def encoder_block(p, filters):
            """Prompt encoder stage."""
            p = conv_block(p, filters)
            p = conv_block(p, filters)
            prompt_skip_connections.append(p)        # save skip
            p = conv_block(p, filters * 2, strides=2)
            return p

        def encoder_block_2(x, p, filters):
            """Image encoder stage conditioned on the SE-gated prompt skip."""
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
        prompt = SeparableConv2D(F[0], (3, 3), padding='same')(prompt)
        for f in F:
            prompt = encoder_block(prompt, f)

        # ── image encoder ─────────────────────────────────────────────
        x = SeparableConv2D(F[0], (3, 3), padding='same')(image)
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
        with tf.GradientTape() as tape:
            y_pred = self.this([z[0], z[2]], training=True)
            loss   = self.loss(z[1], y_pred)

        grads = tape.gradient(loss, self.this.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.this.trainable_variables))
        self.train_loss.update_state(loss)

    def train_epoch(self, train_dataset, batch_size=None, augmenter=None):
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
