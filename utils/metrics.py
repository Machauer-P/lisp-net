import numpy as np
import tensorflow as tf

def to_numpy(x):
    """
    Converts input tensors to a NumPy array.
    Supports NumPy arrays, TensorFlow tensors, and PyTorch (including GPU) tensors.
    """
    # 1. Handle TensorFlow tensors
    if isinstance(x, tf.Tensor):
        return x.numpy()
    
    # 2. Handle PyTorch tensors (using hasattr to avoid a hard dependency on 'torch' at runtime)
    # Tensors on GPU require .detach().cpu().numpy()
    if hasattr(x, 'detach') and hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
        
    # 3. Handle other array-like objects or direct NumPy arrays
    if not isinstance(x, np.ndarray) and hasattr(x, 'numpy'):
        try:
            return x.numpy()
        except Exception:
            pass
            
    return np.asarray(x)

def dice_numpy(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Sørensen–Dice coefficient using NumPy.
    Works with NumPy arrays, TensorFlow tensors, and PyTorch tensors.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        float: Dice score as a NumPy float
    """
    y_true_np = to_numpy(y_true).astype(np.float32)
    y_pred_np = to_numpy(y_pred).astype(np.float32)

    # Flatten
    y_true_f = y_true_np.flatten()
    y_pred_f = y_pred_np.flatten()

    # Dice calculation
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice

def dice_score_tf(y_true, y_pred, smooth=1e-6):
    """
    Legacy TensorFlow implementation of the Dice score.
    Maintains the calculation within the TensorFlow graph (e.g., on GPU).
    """
    if y_true.dtype != tf.float32:
        y_true = tf.cast(y_true, tf.float32)
    if y_pred.dtype != tf.float32:
        y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return dice
