"""
inference/ssf_ablation.py
=========================
Ablation SSF variant: always updates the prompt with the current prediction.

For ablation testing where every slice's prediction becomes the prompt for
the next slice — no threshold condition, no rolling buffer.  This isolates
the contribution of SSF trigger logic and buffer-mean selection.

Slice-level batching
--------------------
Because every slice changes the prompt, slices within a batch can no longer
share a prompt.  Only **tile-level batching** (multiple tiles of one slice)
still works.  Always use ``batch_size=1`` with this controller::

    from inference.ssf_ablation import AlwaysUpdateController
    from inference.inference_volume import VolumeInference

    vol_inf = VolumeInference(
        model_path="training/p_unet_332.keras",
        modality="CT",
        ssf_strategy=None,          # ignored — controller overrides it
        batch_size=1,               # CRITICAL: no slice-level batching
    )
    vol_inf._ssf = AlwaysUpdateController()

Or, equivalently, create the VolumeInference normally and swap the
controller before calling ``run()``::

    vol_inf = VolumeInference(model_path=..., modality=..., batch_size=1)
    vol_inf._ssf = AlwaysUpdateController()

The ``AlwaysUpdateController`` is a drop-in replacement for ``SSFController``
with the same public API (``reset``, ``reset_trigger``, ``update``, ``strategy``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from inference.ssf import BaseSSFStrategy


# ---------------------------------------------------------------------------
# Strategy — always fires, no state
# ---------------------------------------------------------------------------

class AlwaysUpdateStrategy(BaseSSFStrategy):
    """
    SSF strategy that **always** fires — no threshold, no condition.

    Every ``check()`` call returns ``True``, so the prompt is refreshed on
    every single slice.
    """

    def reset(self) -> None:
        """No-op — this strategy has no internal state."""
        pass

    def check(
        self,
        slice_idx: int,
        img_plane_128: np.ndarray,     # (128, 128) float32
        pred_binary_128: np.ndarray,   # (128, 128) float32 binary
        pred_prob_128: np.ndarray,     # (128, 128) float32 sigmoid [0, 1]
        prompt_img_128: np.ndarray,    # (128, 128) float32
    ) -> bool:
        if self.debug:
            print(
                f"[SSF-AlwaysUpdate sl={slice_idx:4d}] "
                f"→ FIRES (unconditional)"
            )
        return True

    @property
    def name(self) -> str:
        return "AlwaysUpdate"


# ---------------------------------------------------------------------------
# Controller — no buffer, always returns current prediction
# ---------------------------------------------------------------------------

class AlwaysUpdateController:
    """
    SSF controller for ablation: always updates the prompt with the **current**
    prediction.  No rolling buffer, no threshold.

    Drop-in replacement for :class:`SSFController`.  Accepts the same
    constructor arguments (``strategy`` and ``buffer_size``) but ignores
    both — the strategy is always ``AlwaysUpdateStrategy`` and no buffer
    is maintained.

    Parameters
    ----------
    strategy : BaseSSFStrategy or None
        Ignored.  The controller always uses ``AlwaysUpdateStrategy``.
    buffer_size : int
        Ignored.  No buffer is kept.
    """

    def __init__(
        self,
        strategy: Optional[BaseSSFStrategy] = None,
        buffer_size: int = 4,
    ):
        self.strategy = AlwaysUpdateStrategy()

    # ------------------------------------------------------------------
    # Public API — mirrors SSFController
    # ------------------------------------------------------------------

    def reset(self, initial_mask_128) -> None:
        """
        Reset for a new propagation direction.

        Parameters
        ----------
        initial_mask_128 : array-like, shape (1, 128, 128, 1)
            Initial prompt mask thumbnail.  Not buffered — just passed
            through for API compatibility.
        """
        if self.strategy is not None:
            self.strategy.reset()

    def reset_trigger(self) -> None:
        """
        Reset only the strategy's trigger state.

        No-op for the ablation — there is no trigger state to reset.
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
    ) -> Tuple[bool, object]:
        """
        Always fires, returning the current binary prediction as the new
        prompt mask.

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
            fired        : always ``True``.
            new_mask_128 : (1, 128, 128, 1) — the current ``pred_binary_128``,
                           returned as-is (no buffer-mean).
        """
        # Convert tensors to numpy for strategy.check() (API compatibility)
        pred_b_np = (
            pred_binary_128.numpy()
            if hasattr(pred_binary_128, 'numpy')
            else np.asarray(pred_binary_128)
        )
        pred_p_np = (
            pred_prob_128.numpy()
            if hasattr(pred_prob_128, 'numpy')
            else np.asarray(pred_prob_128)
        )

        if self.strategy is not None:
            self.strategy.check(
                slice_idx       = slice_idx,
                img_plane_128   = img_plane_128,
                pred_binary_128 = pred_b_np[0, ..., 0],   # (128, 128)
                pred_prob_128   = pred_p_np[0, ..., 0],   # (128, 128)
                prompt_img_128  = prompt_img_128,
            )

        # Always fire — return the current prediction as the new prompt mask.
        # Keep the original type (TF tensor or ndarray) for drop-in compatibility.
        return True, pred_binary_128
