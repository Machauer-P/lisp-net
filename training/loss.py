import tensorflow as tf

class DiceBCELoss(tf.keras.losses.Loss):
    """
    nnUNet-inspired Loss Function for Binary Segmentation.
    
    Combines Binary Cross-Entropy (BCE) with a Soft Batch Dice Loss.
    Unlike standard per-sample Dice, computing the Dice coefficient across
    the entire batch stabilizes the gradients significantly when dealing with
    extreme class imbalances or when many slices in a batch contain no foreground
    (which otherwise causes Dice score to crash to 0 and gradients to explode / vanish).
    
    This is highly effective at preventing the "predict all zeros" death spiral.
    """
    def __init__(self, smooth=1e-5, bce_weight=1.0, dice_weight=1.0, name='dice_bce_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # Standard BCE, returning the scalar mean over the batch
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        # 1. Binary Crossentropy Loss
        bce_loss = self.bce(y_true, y_pred)
        
        # 2. Batch Dice Loss
        # Cast to float32 and flatten entirely to compute global batch intersection
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        sum_y_true = tf.reduce_sum(y_true_f)
        sum_y_pred = tf.reduce_sum(y_pred_f)
        
        dice = (2.0 * intersection + self.smooth) / (sum_y_true + sum_y_pred + self.smooth)
        dice_loss = 1.0 - dice
        
        # 3. Combine
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
