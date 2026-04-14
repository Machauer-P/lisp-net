"""
Prompt U-Net Model

This module contains the definition of the Prompt U-Net utilizing TensorFlow and Keras.
It is designed to take an image and an accompanying spatial prompt as inputs,
fusing their encodings, and finally decoding to a probability map for segmentation.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Conv2D


class PromptUNet:
    """
    Prompt U-Net model class combining prompt encoding, image encoding, and decoding.
    
    The architecture is a dual-encoder UNet. One branch encodes the prompt masks,
    extracting multi-scale features. The other branch encodes the image while
    conditioning on the prompt's intermediate feature maps via skip connections.
    Finally, a decoder reconstructs the spatial resolution using skip connections
    from the combined encoding branch.

    Attributes:
        height (int): The spatial height of the inputs.
        width (int): The spatial width of the inputs.
        loss (callable): Loss function used for training (default: binary_crossentropy).
        train_loss (tf.keras.metrics.Mean): A metric object for computing average training loss.
        this (tf.keras.Model): The instantiated tf.keras functional model.
        optimizer (tf.optimizers.Optimizer): The optimizer to be used (can be set externally, e.g., LossScaleOptimizer).
    """

    def __init__(self, height=128, width=128):
        """
        Initializes the model architecture.

        Args:
            height (int): Height dimension for the network inputs.
            width (int): Width dimension for the network inputs.
        """
        self.height = height
        self.width = width
        self.loss = tf.losses.binary_crossentropy
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.this = self.build()
        self.optimizer = None  # Expected to be assigned by the training loop/caller


    def build(self) -> tf.keras.Model:
        """
        Constructs the Keras functional API model containing the dual encoder and decoder.

        Returns:
            tf.keras.Model: The compiled architecture taking two arguments `[image, prompt]`
            and outputting the predicted sigmoid probability scalar map.
        """
        inputs = [
            tf.keras.Input(shape=(self.height, self.width, 1), name='image'),
            tf.keras.Input(shape=(self.height, self.width, 2), name='prompt')
        ]

        image = inputs[0]
        prompt = inputs[1]

        prompt_skip_connections = []
        skip_connections = []

        def conv_block(inp, filters, padding='same', activation='leaky_relu', dropout_rate=0.1, **kwargs):
            """
            Standard convolution block encompassing Separable Conv2D, BatchNormalization, 
            LeakyReLU activation, and Dropout.
            """
            inp = layers.SeparableConv2D(filters, (3, 3), padding=padding, **kwargs)(inp)
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            return inp

        def conv_block_prompt(x, p, filters, padding='same', activation='leaky_relu', dropout_rate=0.1):           
            """
            Convolution block that adds the prompt's condition to the main encoding branch.
            """
            p = conv_block(p, filters)
            x = layers.SeparableConv2D(filters, (3, 3), padding=padding)(x)
            x = layers.Add()([x, p])
            x = layers.Dropout(dropout_rate)(x)
            return x

        def conv_block_up(inp, filters, padding='same', activation='leaky_relu', dropout_rate=0.1, **kwargs):
            """
            Upsampling block applying math-based scaling followed by SeparableConv convolution.
            """
            inp = layers.BatchNormalization()(inp)
            inp = layers.LeakyReLU()(inp)
            inp = layers.Dropout(dropout_rate)(inp)
            
            # 1. Scale up spatially using math (no parameters, very fast)
            inp = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(inp)     
            # 2. Smooth and adjust channels
            inp = layers.SeparableConv2D(filters, (3, 3), padding=padding)(inp)
            
            return inp

        def encoder_block(p, filters, padding='same', activation='leaky_relu'):
            """
            Encoding stage specifically for the prompt tensor branch.
            Draws out the skip connection variables for cross-branch conditioning.
            """
            p = conv_block(p, filters, padding, activation)
            p = conv_block(p, filters, padding, activation)
            prompt_skip_connections.append(p)
            p = conv_block(p, filters * 2, padding, strides=2)
            return p

        def encoder_block_2(x, p, filters, padding='same'):
            """
            Encoding stage for the image tensor branch conditioned on the prompt.
            Uses strided convolution instead of pooling to extract spatially reduced structures.
            """
            x = conv_block_prompt(x, p, filters)        
            skip_connections.append(x)
            x = conv_block(x, filters * 2, padding, strides=2)
            return x

        def decoder_block(inp, concat_layer, filters, padding='same', dropout_rate=0.1):
            """
            Decoding standard block that combines the un-pooled feature map and the matched
            skip connections before final convolutions limit spatial resolution issues.
            """
            x = conv_block_up(inp, filters, padding, strides=2)
            x = concatenate([x, concat_layer])      
            x = conv_block(x, filters, padding, dropout_rate=dropout_rate)
            return x

        # --- Encoding prompt ---
        prompt = layers.SeparableConv2D(32, (3, 3), padding="same")(prompt)
        prompt = encoder_block(prompt, 32)
        prompt = encoder_block(prompt, 64)
        prompt = encoder_block(prompt, 128)
        prompt = encoder_block(prompt, 256)
        prompt = encoder_block(prompt, 512)

        # --- Encoding x (with conditioning on prompt) ---
        x = layers.SeparableConv2D(32, (3, 3), padding="same")(image)
        x = encoder_block_2(x=x, p=prompt_skip_connections.pop(0), filters=32) 
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 64)
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 128)
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 256)
        x = encoder_block_2(x, prompt_skip_connections.pop(0), 512)

        # --- Middle part ---
        x = conv_block(x, 512, dropout_rate=0.2)  # Higher Dropout-Rate in bottleneck

        # --- Decoding / Upsampling (with skip connections) ---
        x = decoder_block(x, skip_connections.pop(), 512)
        x = decoder_block(x, skip_connections.pop(), 256)
        x = decoder_block(x, skip_connections.pop(), 128)
        x = decoder_block(x, skip_connections.pop(), 64)
        x = decoder_block(x, skip_connections.pop(), 32)

        # Output projection map
        output = Conv2D(1, 1)(x)                               
        output = tf.keras.activations.sigmoid(output)          

        return tf.keras.Model(inputs=inputs, outputs=output)

    @tf.function
    def train_step(self, z):
        """
        Executes a single optimization step using automatic differentiation with gradient tape.

        Args:
            z (tuple): A batch structured as (image, label, prompt).
                - image (tf.Tensor): Batched input images (None, height, width, 1)
                - label (tf.Tensor): Batched ground-truth mask (None, height, width, 1)
                - prompt (tf.Tensor): Batched prompts mapping to shape (None, height, width, 2)
        """
        with tf.GradientTape() as tape:
            # z[0] = image/x, z[1] = label/y, z[2] = prompt/p
            y_pred = self.this([z[0], z[2]], training=True)  
            loss = self.loss(z[1], y_pred) 
            scaled_loss = self.optimizer.scale_loss(loss)  # For mixed precision context

        # Calculate backpropagation gradients and unscale according to the LossScaleOptimizer
        scaled_gradients = tape.gradient(scaled_loss, self.this.trainable_variables)
        self.optimizer.apply_gradients(zip(scaled_gradients, self.this.trainable_variables))
        
        # Track training loss across batch executions
        self.train_loss.update_state(loss)

    def train_epoch(self, train_dataset, batch_size=None, augmenter=None):
        """
        Runs the training subroutine across random subsets of the dataset.

        Args:
            train_dataset (tf.data.Dataset): The dataset formatted as tuples (image, label, prompt).
            batch_size (int, optional): Batch size utilized during mapping. If None, assumes already batched.
            augmenter (callable, optional): Data augmenter callable mapped dynamically during iterations.
        """
        if augmenter is not None or batch_size is not None:
            train_dataset = train_dataset.shuffle(256)
            if augmenter is not None:
                 train_dataset = train_dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
            if batch_size is not None:
                 train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        for z in train_dataset:
            self.train_step(z)
