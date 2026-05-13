import numpy as np
import sys

def to_numpy(x):
    """
    Converts input tensors to a NumPy array.
    Supports NumPy arrays, TensorFlow tensors, and PyTorch (including GPU) tensors.
    """
    # 1. Handle TensorFlow tensors
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
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
    import tensorflow as tf
    if y_true.dtype != tf.float32:
        y_true = tf.cast(y_true, tf.float32)
    if y_pred.dtype != tf.float32:
        y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return dice


# ---------------------------------------------------------------------------
# 3-D / windowed Dice helpers (moved from benchmark_nninteractive/3d_test.ipynb)
# ---------------------------------------------------------------------------

def volumetric_dice(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient over the entire 3-D binary volume.

    Both inputs are binarised (any non-zero value = foreground).
    Returns 1.0 when both volumes are completely empty (correct prediction).

    Args:
        y_true : np.ndarray or tensor, shape (D, H, W) or any matching shape.
        y_pred : np.ndarray or tensor, same shape as y_true.
        smooth : float — Laplace smoothing to avoid 0/0.

    Returns:
        float in [0, 1].
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    A = (y_true != 0).astype(np.bool_)
    B = (y_pred != 0).astype(np.bool_)
    inter = np.sum(A & B)
    sumA  = np.sum(A)
    sumB  = np.sum(B)
    if sumA == 0 and sumB == 0:
        return 1.0
    return float((2.0 * inter + smooth) / (sumA + sumB + smooth))


def dice_window_nn(y_true_3d, y_pred_3d, axis, center_idx, window=10, smooth=1e-6):
    """
    Mean slice-wise Dice in a ±`window` band around `center_idx` along `axis`.

    Used to measure how well nnInteractive reconstructed the neighbourhood of
    the interaction prompt slice.

    Args:
        y_true_3d  : np.ndarray or tensor, shape (D, H, W).
        y_pred_3d  : np.ndarray or tensor, shape (D, H, W).
        axis       : int — the axis along which the prompt was given (0, 1, or 2).
        center_idx : int — the prompt slice index.
        window     : int — half-width of the evaluation band.
        smooth     : float — Laplace smoothing.

    Returns:
        float — mean Dice over slices in [center_idx-window, center_idx+window].
    """
    y_true_3d = to_numpy(y_true_3d).astype(np.float32)
    y_pred_3d = to_numpy(y_pred_3d).astype(np.float32)

    # Move the chosen axis to the front so slicing is uniform
    y_true_3d = np.moveaxis(y_true_3d, axis, 0)
    y_pred_3d = np.moveaxis(y_pred_3d, axis, 0)

    num_slices = y_true_3d.shape[0]
    start = max(center_idx - window, 0)
    end   = min(center_idx + window + 1, num_slices)

    y_true_win = y_true_3d[start:end]
    y_pred_win = y_pred_3d[start:end]

    n = y_true_win.shape[0]
    dice_scores = [dice_numpy(y_true_win[i], y_pred_win[i], smooth) for i in range(n)]
    
    return float(np.mean(dice_scores))


def dice_window_prompt(y_true_3d, y_pred_3d, forward_idxs, window=10, smooth=1e-6):
    """
    Mean slice-wise Dice in a ±`window` band centred just before the forward
    indices start (i.e. around the last backward-propagation slice).

    Used to evaluate Prompt-UNet quality near the handover point from backward
    to forward propagation.

    Args:
        y_true_3d   : np.ndarray or tensor, shape (slices, H, W).  Arranged in
                      evaluation order (backward slices reversed + prompt + forward).
        y_pred_3d   : np.ndarray or tensor, same shape.
        forward_idxs: list — the forward slice indices returned by VolumeInference.
                      Only its *length* matters here (used to compute the centre).
        window      : int — half-width of the evaluation band.
        smooth      : float — Laplace smoothing.

    Returns:
        float — mean Dice over slices in the window.
    """
    y_true_3d = to_numpy(y_true_3d).astype(np.float32)
    y_pred_3d = to_numpy(y_pred_3d).astype(np.float32)

    num_slices = y_true_3d.shape[0]
    center_idx = (num_slices - len(forward_idxs)) - 1

    start = max(center_idx - window, 0)
    end   = min(center_idx + window + 1, num_slices)

    y_true_win = y_true_3d[start:end]
    y_pred_win = y_pred_3d[start:end]

    n = y_true_win.shape[0]
    dice_scores = [dice_numpy(y_true_win[i], y_pred_win[i], smooth) for i in range(n)]
    
    return float(np.mean(dice_scores))
