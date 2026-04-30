"""
Prompt U-Net Optimizer and Learning Rate Scheduler Configuration.
(Legacy — used only by p_unet_292.ipynb)

This module provides a class to easily instantiate the learning rate 
schedule and the optimizer configured specifically for the Prompt U-Net 
training process, ensuring mixed-precision support.
"""

import tensorflow as tf

class PromptUNetOptimizer:
    """
    Optimizer and Learning Rate Scheduler configuration for Prompt U-Net.
    
    Implements a Cosine Decay learning rate schedule with a warmup phase,
    and returns an Adam optimizer wrapped in a mixed-precision LossScaleOptimizer.
    
    Attributes:
        epochs (int): Total number of training epochs.
        batch_size (int): Size of the batches used in training.
        dp_training (int): Total number of data points for training per epoch.
        warmup_epochs (int): Number of epochs dedicated to warming up the learning rate.
        initial_learning_rate (float): Starting learning rate before warmup.
        warmup_target (float): Peak learning rate after warmup.
        alpha (float): Minimum learning rate at the end of decay as a fraction of `warmup_target`.
    """
    def __init__(
        self, 
        epochs=4000, 
        batch_size=128, 
        dp_training=3500, 
        warmup_epochs=50,
        initial_learning_rate=1e-6, 
        warmup_target=1e-3, 
        alpha=0.01
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.dp_training = dp_training
        self.warmup_epochs = warmup_epochs
        
        self.initial_learning_rate = initial_learning_rate
        self.warmup_target = warmup_target
        self.alpha = alpha
        
        # Calculate steps
        self.steps_per_epoch = self.dp_training // self.batch_size
        self.total_steps = self.steps_per_epoch * self.epochs
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.decay_steps = self.total_steps - self.warmup_steps

    def get_lr_schedule(self) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Generates and returns the CosineDecay learning rate schedule.
        
        Returns:
            tf.keras.optimizers.schedules.CosineDecay: The configured learning rate schedule.
        """
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.initial_learning_rate,
            warmup_target=self.warmup_target,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            alpha=self.alpha
        )
        return lr_schedule

    def get_optimizer(self) -> tf.keras.mixed_precision.LossScaleOptimizer:
        """
        Returns the optimized setup ready to be assigned to the Prompt U-Net model.
        
        Returns:
            tf.keras.mixed_precision.LossScaleOptimizer: An Adam optimizer utilizing 
            the computed LR schedule, wrapped for mixed-float16 precision tracking.
        """
        lr_schedule = self.get_lr_schedule()
        base_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Wrap it for mixed precision gradient scaling
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
        return optimizer
