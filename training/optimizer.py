"""
Prompt U-Net Optimizer and Learning Rate Scheduler Configuration.

Implements the three-phase WarmupFlatCosineDecay schedule used in v310+:
  1. Linear warmup  : initial_lr → peak_lr over `warmup_epochs` epochs
  2. Flat plateau   : holds at peak_lr for `flat_epochs` epochs
  3. Cosine decay   : peak_lr → min_lr over the remaining epochs

Returns a plain Adam optimizer (float32). For the legacy mixed-precision
variant used by p_unet_292, see p_unet_292_optimizer.py.
"""

import math
import tensorflow as tf


class WarmupFlatCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Three-phase learning rate schedule used in Prompt U-Net v310+.

    Phases (all measured in gradient steps, not epochs):
      1. Linear warmup  : initial_lr → peak_lr over `warmup_steps` steps.
      2. Flat plateau   : stays at peak_lr for `flat_steps` steps.
      3. Cosine decay   : peak_lr → min_lr over `decay_steps` steps.

    Args:
        warmup_steps (int): Number of gradient steps for the warmup phase.
        flat_steps (int):   Number of gradient steps for the flat plateau.
        decay_steps (int):  Number of gradient steps for the cosine decay.
        initial_lr (float): Starting LR at step 0 (before warmup). Default 1e-6.
        peak_lr (float):    Target LR at end of warmup / during plateau. Default 1e-3.
        alpha (float):      Minimum LR as a fraction of peak_lr. Default 0.01
                            (i.e. final LR = 1e-5 when peak_lr = 1e-3).
    """

    def __init__(
        self,
        warmup_steps: int,
        flat_steps: int,
        decay_steps: int,
        initial_lr: float = 1e-6,
        peak_lr: float = 1e-3,
        alpha: float = 0.01,
    ):
        super().__init__()
        self.warmup_steps = float(warmup_steps)
        self.flat_steps   = float(flat_steps)
        self.decay_steps  = float(decay_steps)
        self.initial_lr   = initial_lr
        self.peak_lr      = peak_lr
        self.min_lr       = alpha * peak_lr

    def __call__(self, step):
        step     = tf.cast(step, tf.float32)
        flat_end = self.warmup_steps + self.flat_steps

        # Phase 1: linear warmup
        warmup_frac = tf.minimum(step / self.warmup_steps, 1.0)
        warmup_lr   = self.initial_lr + (self.peak_lr - self.initial_lr) * warmup_frac

        # Phase 3: cosine decay
        decay_step = step - flat_end
        decay_frac = tf.minimum(tf.maximum(decay_step / self.decay_steps, 0.0), 1.0)
        cosine_lr  = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                         1.0 + tf.cos(math.pi * decay_frac))

        return tf.where(step < self.warmup_steps, warmup_lr,
               tf.where(step < flat_end,          self.peak_lr, cosine_lr))

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "flat_steps":   self.flat_steps,
            "decay_steps":  self.decay_steps,
            "initial_lr":   self.initial_lr,
            "peak_lr":      self.peak_lr,
            "min_lr":       self.min_lr,
        }


class PromptUNetOptimizer:
    """
    Convenience wrapper that builds the WarmupFlatCosineDecay schedule
    and a plain Adam optimizer for Prompt U-Net v310+ (pure float32).

    All epoch-based arguments are converted to gradient steps internally
    using `steps_per_epoch = dp_training // batch_size`.

    Args:
        epochs (int):         Total training epochs. Default 4000.
        batch_size (int):     Mini-batch size. Default 128.
        dp_training (int):    Training buffer size (data points per refresh). Default 10000.
        warmup_epochs (int):  Epochs for the linear warmup phase. Default 50.
        flat_epochs (int):    Epochs for the flat plateau phase. Default 1500.
        initial_lr (float):   LR at step 0. Default 1e-6.
        peak_lr (float):      Peak / plateau LR. Default 1e-3.
        alpha (float):        Min LR fraction of peak_lr. Default 0.01.
    """

    def __init__(
        self,
        epochs: int         = 4000,
        batch_size: int     = 128,
        dp_training: int    = 10000,
        warmup_epochs: int  = 50,
        flat_epochs: int    = 1500,
        initial_lr: float   = 1e-6,
        peak_lr: float      = 1e-3,
        alpha: float        = 0.01,
    ):
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.dp_training   = dp_training
        self.warmup_epochs = warmup_epochs
        self.flat_epochs   = flat_epochs
        self.initial_lr    = initial_lr
        self.peak_lr       = peak_lr
        self.alpha         = alpha

        self.steps_per_epoch = dp_training // batch_size
        self.warmup_steps    = warmup_epochs * self.steps_per_epoch
        self.flat_steps      = flat_epochs   * self.steps_per_epoch
        self.decay_steps     = (epochs - warmup_epochs - flat_epochs) * self.steps_per_epoch

    def get_lr_schedule(self) -> WarmupFlatCosineDecay:
        """Returns the configured WarmupFlatCosineDecay schedule."""
        return WarmupFlatCosineDecay(
            warmup_steps = self.warmup_steps,
            flat_steps   = self.flat_steps,
            decay_steps  = self.decay_steps,
            initial_lr   = self.initial_lr,
            peak_lr      = self.peak_lr,
            alpha        = self.alpha,
        )

    def get_optimizer(self) -> tf.keras.optimizers.Adam:
        """
        Returns a plain Adam optimizer with the WarmupFlatCosineDecay schedule.

        Pure float32 — no LossScaleOptimizer. Assign directly to model.optimizer.
        """
        return tf.keras.optimizers.Adam(learning_rate=self.get_lr_schedule())
