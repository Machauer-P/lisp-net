"""
inference/ssf.py
================
Pluggable Self-Supervised Feedback (SSF) strategies for Prompt-UNet slice
propagation.

SSF detects when the current prompt has drifted out of alignment with the
anatomy being traversed and refreshes it from a rolling buffer of recent
predictions (buffer-minimum strategy).

Three strategies are provided, each implementing ``BaseSSFStrategy``:

    RelativeSSIMStrategy
        Fires when ``(start_ssim - ssim_k) / start_ssim >= threshold``.
        Normalises the SSIM drop to the starting similarity so that the same
        threshold works across anatomy/modalities/datasets.
        Typical threshold: 0.20 – 0.30.

    MaskDiceStrategy
        Fires when ``dice(pred_{k-1}, pred_k) < threshold``.
        Completely image- and modality-agnostic: detects self-consistency
        collapse in the model's own output. The most universally stable
        strategy.
        Typical threshold: 0.35 – 0.50.

    ConfidenceDropStrategy
        Fires when mean sigmoid probability over predicted foreground drops
        below ``start_confidence * (1 - drop_fraction)``.
        Sensitive to gradual confidence erosion before the binary mask
        collapses.
        Typical drop_fraction: 0.25 – 0.40.

Usage
-----
    from inference.ssf import SSFController, RelativeSSIMStrategy

    ctrl = SSFController(strategy=RelativeSSIMStrategy(threshold=0.40),
                         buffer_size=4)

    # At the start of each propagation direction:
    ctrl.reset(initial_mask_128)   # (1, 128, 128, 1) binary tensor

    # After each slice prediction:
    fired, new_mask = ctrl.update(
        slice_idx       = i,
        img_plane_128   = img_128,     # (128, 128) float32 — normalized
        pred_binary_128 = pred_128,    # (1, 128, 128, 1) float32 binary tensor
        pred_prob_128   = prob_128,    # (1, 128, 128, 1) float32 sigmoid [0,1] tensor
        prompt_img_128  = p_img_128,   # (128, 128) float32 — frozen prompt image
    )
    if fired:
        # new_mask is (1, 128, 128, 1) binary TF tensor — the buffer-min refresh
        ...

Disabling SSF
-------------
Pass ``strategy=None`` (or omit it) to ``SSFController`` to disable SSF
entirely with no performance overhead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.image import ssim as tf_ssim


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSSFStrategy(ABC):
    """
    Abstract base class for SSF trigger strategies.

    Each strategy is stateful (stores start references across slices) and
    must reset itself when the propagation direction switches or when SSF fires.

    Enable per-slice console diagnostics by setting ``instance.debug = True``.
    """

    debug: bool = False

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new propagation direction or after SSF fires."""
        ...

    @abstractmethod
    def check(
        self,
        slice_idx: int,
        img_plane_128: np.ndarray,     # (128, 128) float32 normalized image
        pred_binary_128: np.ndarray,   # (128, 128) float32 binary prediction
        pred_prob_128: np.ndarray,     # (128, 128) float32 raw sigmoid [0, 1]
        prompt_img_128: np.ndarray,    # (128, 128) float32 frozen prompt image
    ) -> bool:
        """Return True if SSF should fire on this slice."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Strategy 1 — Relative SSIM drop
# ---------------------------------------------------------------------------

class RelativeSSIMStrategy(BaseSSFStrategy):
    """
    Fires when SSIM has dropped by more than ``threshold`` *relative* to its
    starting value:

        (start_ssim - ssim_k) / start_ssim  >=  threshold

    Unlike the legacy absolute SSIM check, the relative formulation is
    invariant to the baseline similarity, which varies significantly by
    anatomy, modality, and dataset.

    Parameters
    ----------
    threshold : float
        Relative SSIM drop fraction that triggers SSF.  Default 0.40.
        A value of 0.40 means "fire when SSIM has degraded by 40% of its
        initial value".
        
        NOTE: This default parameter was found via an evaluation on the training data.
        For other datasets, other settings might work better, and there is always
        the option to just turn SSF off.
    """

    def __init__(self, threshold: float = 0.40):
        self.threshold      = threshold
        self._start_ssim: Optional[float] = None

    def reset(self) -> None:
        self._start_ssim = None

    def check(
        self,
        slice_idx: int,
        img_plane_128: np.ndarray,
        pred_binary_128: np.ndarray,
        pred_prob_128: np.ndarray,
        prompt_img_128: np.ndarray,
    ) -> bool:
        ssim_k = _compute_ssim_joint(img_plane_128, prompt_img_128)
        if self._start_ssim is None:
            self._start_ssim = ssim_k

        relative_drop = (self._start_ssim - ssim_k) / (self._start_ssim + 1e-8)
        fires = (self.threshold > 0
                 and relative_drop >= self.threshold
                 and relative_drop < 1.0)

        if self.debug:
            print(
                f"[SSF-RelSSIM  sl={slice_idx:4d}] "
                f"ssim={ssim_k:.4f} start={self._start_ssim:.4f} "
                f"rel_drop={relative_drop:.4f} thr={self.threshold:.2f} "
                f"→ {'FIRES' if fires else 'ok'}"
            )
        return fires

    @property
    def name(self) -> str:
        return f"RelativeSSIM(t={self.threshold})"


# ---------------------------------------------------------------------------
# Strategy 2 — Consecutive mask Dice
# ---------------------------------------------------------------------------

class MaskDiceStrategy(BaseSSFStrategy):
    """
    Fires when consecutive predicted masks disagree:

        dice(pred_{k-1}, pred_k)  <  threshold

    Completely image- and modality-agnostic: only examines whether the model's
    own output is self-consistent from slice to slice.  A sudden Dice collapse
    signals prompt misalignment without any image statistics.

    Parameters
    ----------
    threshold : float
        Minimum consecutive-mask Dice considered stable.  Below this, SSF
        fires.  Default 0.40.
    """

    def __init__(self, threshold: float = 0.40):
        self.threshold              = threshold
        self._prev_pred: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_pred = None

    def check(
        self,
        slice_idx: int,
        img_plane_128: np.ndarray,
        pred_binary_128: np.ndarray,
        pred_prob_128: np.ndarray,
        prompt_img_128: np.ndarray,
    ) -> bool:
        if self._prev_pred is None:
            # No previous slice yet — cannot compare; seed reference instead.
            self._prev_pred = pred_binary_128.copy()
            return False

        dice = _binary_dice(self._prev_pred, pred_binary_128)
        fires = dice < self.threshold

        if self.debug:
            print(
                f"[SSF-MaskDice sl={slice_idx:4d}] "
                f"consec_dice={dice:.4f} thr={self.threshold:.2f} "
                f"→ {'FIRES' if fires else 'ok'}"
            )

        self._prev_pred = pred_binary_128.copy()
        return fires

    @property
    def name(self) -> str:
        return f"MaskDice(t={self.threshold})"


# ---------------------------------------------------------------------------
# Strategy 3 — Model confidence drop
# ---------------------------------------------------------------------------

class ConfidenceDropStrategy(BaseSSFStrategy):
    """
    Fires when mean sigmoid probability (model confidence) drops by more than
    ``drop_fraction`` relative to its value at the first evaluated slice:

        mean_prob_k  <  start_confidence * (1 - drop_fraction)

    Operates on raw sigmoid probabilities (before thresholding), so it is
    sensitive to gradual confidence erosion that may not yet be visible in the
    binary mask.

    Parameters
    ----------
    drop_fraction : float
        Relative confidence drop that triggers SSF.  Default 0.30.
    min_foreground_fraction : float
        Minimum fraction of 128×128 pixels that must be predicted as foreground
        before the confidence check runs.  Prevents false-positives on
        near-empty predictions.  Default 0.005 (0.5 %).
    """

    def __init__(
        self,
        drop_fraction: float = 0.05,
        min_foreground_fraction: float = 0.005,
    ):
        self.drop_fraction          = drop_fraction
        self.min_foreground_fraction = min_foreground_fraction
        self._start_confidence: Optional[float] = None

    def reset(self) -> None:
        self._start_confidence = None

    def check(
        self,
        slice_idx: int,
        img_plane_128: np.ndarray,
        pred_binary_128: np.ndarray,
        pred_prob_128: np.ndarray,
        prompt_img_128: np.ndarray,
    ) -> bool:
        fg_mask     = pred_binary_128 > 0.5
        fg_fraction = fg_mask.mean()

        if fg_fraction < self.min_foreground_fraction:
            if self.debug:
                print(
                    f"[SSF-Confidence sl={slice_idx:4d}] "
                    f"fg={fg_fraction:.4f} → skip (near-empty)"
                )
            return False

        confidence_k = (float(pred_prob_128[fg_mask].mean())
                        if fg_mask.any() else float(pred_prob_128.mean()))

        if self._start_confidence is None:
            self._start_confidence = confidence_k

        floor = self._start_confidence * (1.0 - self.drop_fraction)
        fires = confidence_k < floor

        if self.debug:
            print(
                f"[SSF-Confidence sl={slice_idx:4d}] "
                f"conf={confidence_k:.4f} start={self._start_confidence:.4f} "
                f"floor={floor:.4f} drop_frac={self.drop_fraction:.2f} "
                f"→ {'FIRES' if fires else 'ok'}"
            )
        return fires

    @property
    def name(self) -> str:
        return f"ConfidenceDrop(df={self.drop_fraction})"


# ---------------------------------------------------------------------------
# SSF Controller — buffer management + strategy bridge
# ---------------------------------------------------------------------------

class SSFController:
    """
    Orchestrates SSF buffer management and delegates the trigger decision to a
    pluggable ``BaseSSFStrategy``.

    Parameters
    ----------
    strategy : BaseSSFStrategy or None
        Which SSF strategy to use.  ``None`` disables SSF entirely.
    buffer_size : int
        Rolling buffer depth for the buffer-minimum prompt refresh.  Default 4.
        
        NOTE: This default parameter was found via an evaluation on the training data.
        For other datasets, other settings might work better, and there is always
        the option to just turn SSF off.
    """

    def __init__(
        self,
        strategy: Optional[BaseSSFStrategy] = None,
        buffer_size: int = 4,
    ):
        self.strategy = strategy
        self._buffer  = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initial_mask_128) -> None:
        """
        Reset for a new propagation direction (or a new run).

        Clears the rolling buffer, seeds it with the initial prompt mask,
        and resets the strategy's internal trigger state.

        Parameters
        ----------
        initial_mask_128 : array-like, shape (1, 128, 128, 1)
            Initial prompt mask thumbnail (TF tensor or numpy array).
        """
        self._buffer.clear()
        self._buffer.append(initial_mask_128)
        if self.strategy is not None:
            self.strategy.reset()

    def reset_trigger(self) -> None:
        """
        Reset only the strategy's trigger state without touching the buffer.
        Called when IFL substitutes GT and we want to restart the SSIM/Dice
        reference without discarding buffered predictions.
        """
        if self.strategy is not None:
            self.strategy.reset()

    def update(
        self,
        slice_idx: int,
        img_plane_128: np.ndarray,     # (128, 128) float32 normalized
        pred_binary_128,               # (1, 128, 128, 1) TF tensor or ndarray
        pred_prob_128,                 # (1, 128, 128, 1) TF tensor or ndarray
        prompt_img_128: np.ndarray,    # (128, 128) float32 frozen prompt image
    ) -> Tuple[bool, Optional[object]]:
        """
        Feed one predicted slice into the SSF controller.

        Appends the binary prediction to the rolling buffer, then asks the
        strategy whether SSF should fire.  If yes, computes the buffer-minimum
        refresh mask, clears the buffer (seeding it with the refresh mask),
        and resets the strategy.

        Parameters
        ----------
        slice_idx       : volume slice index (for debug logging).
        img_plane_128   : (128, 128) float32 normalized image plane.
        pred_binary_128 : (1, 128, 128, 1) binary prediction thumbnail.
        pred_prob_128   : (1, 128, 128, 1) raw sigmoid probability thumbnail.
        prompt_img_128  : (128, 128) float32 frozen prompt image channel.

        Returns
        -------
        (fired, new_mask_128)
            fired        : True if SSF triggered.
            new_mask_128 : (1, 128, 128, 1) buffer-min refresh mask TF tensor,
                           or None if not fired.
        """
        self._buffer.append(pred_binary_128)

        if self.strategy is None:
            return False, None

        # Convert tensors to numpy for strategy.check()
        pred_b_np = (pred_binary_128.numpy()
                     if hasattr(pred_binary_128, 'numpy') else np.asarray(pred_binary_128))
        pred_p_np = (pred_prob_128.numpy()
                     if hasattr(pred_prob_128, 'numpy') else np.asarray(pred_prob_128))

        fired = self.strategy.check(
            slice_idx       = slice_idx,
            img_plane_128   = img_plane_128,
            pred_binary_128 = pred_b_np[0, ..., 0],   # (128, 128)
            pred_prob_128   = pred_p_np[0, ..., 0],   # (128, 128)
            prompt_img_128  = prompt_img_128,
        )

        if fired:
            new_mask = self._buffer_min()
            # Reset after fire: clear buffer, seed with refresh mask, reset trigger
            self._buffer.clear()
            self._buffer.append(new_mask)
            self.strategy.reset()
            return True, new_mask

        return False, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _buffer_min(self):
        """Return buffer-minimum mask as (1, 128, 128, 1) binary TF tensor."""
        stacked = tf.stack(list(self._buffer), axis=0)   # (N, 1, 128, 128, 1)
        min_val = tf.reduce_min(stacked, axis=0)          # (1, 128, 128, 1)
        return tf.math.sign(min_val)


# ---------------------------------------------------------------------------
# Module-level helpers shared by strategies
# ---------------------------------------------------------------------------

def _compute_ssim_joint(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute SSIM between two (128, 128) images using **joint** min/max
    rescaling so that relative intensity differences between the two images
    are preserved.

    Independent per-image rescaling would equalize brightness globally and
    destroy the inter-slice structural variation that makes SSIM a useful
    SSF trigger signal.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    global_min = min(a.min(), b.min())
    global_max = max(a.max(), b.max())
    scale = global_max - global_min + 1e-8
    a = (a - global_min) / scale
    b = (b - global_min) / scale
    a_t = tf.constant(a[np.newaxis, ..., np.newaxis])  # (1, 128, 128, 1)
    b_t = tf.constant(b[np.newaxis, ..., np.newaxis])
    score = tf_ssim(a_t, b_t, max_val=1.0)
    return float(tf.squeeze(score).numpy())


def _binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient between two binary (H, W) arrays."""
    a = (a > 0.5).astype(np.float32)
    b = (b > 0.5).astype(np.float32)
    intersection = (a * b).sum()
    denom = a.sum() + b.sum()
    if denom < 1e-6:
        return 1.0  # both empty → perfectly self-consistent
    return float(2.0 * intersection / denom)
