"""
inference/volume_inference.py
=============================
3-D slice-propagation inference for Prompt-UNet.

Two public classes:

    VolumeInference
        Self-Supervised Feedback (SSF) only.  No ground-truth required.
        For inference on new, unannotated volumes.

    InteractiveFeedbackLoop(VolumeInference)
        SSF + Interactive Feedback Loop (IFL).  Requires ground-truth volume.
        When the per-slice Dice drops below `gt_dice_threshold`, the GT slice
        is substituted for the predicted mask — exactly matching the original
        3d_test.ipynb behaviour.  The resulting `user_interacts_idx` list is
        forwarded to nnInteractive for a fair same-interaction comparison.

Normalization — automatic version-based selection
-------------------------------------------------
The model filename is parsed for a version number:
  • version <  292  →  slice-wise ``min_max_norm()``   (legacy, TF-based)
  • version >= 292  →  volume-based ``universal_normalization()`` (z-score)

Override at any time via the ``normalization`` constructor argument:
    'auto'      — detect from filename (default)
    'universal' — force z-score (v292+ convention)
    'legacy'    — force slice-wise min-max (pre-v292 convention)

Performance — graph-mode mini-batch inference
---------------------------------------------
The model is JIT-compiled with ``tf.function`` and a fixed input signature
(dynamic batch dimension, fixed 128×128 spatial size).  By default, slices
are predicted in mini-batches of 3.  This gives a 2–3× throughput improvement
over single-slice eager calls on GPU.

SSF / IFL rollback with mini-batching
--------------------------------------
After predicting a mini-batch, slices are accepted one at a time:

1. The prediction for slice k is added to the rolling buffer.
2. (IFL only) If Dice(pred_k, gt_k) < threshold → slice k is corrected with GT,
   the prompt p is updated, and slices k+1 … batch_end are rolled back
   (they were predicted with the wrong p) and re-queued.
3. The SSIM-based SSF check is applied to slice k.  If it fires → the prompt
   is refreshed from the buffer minimum for slice k+1 onwards, and the same
   rollback of k+1 … batch_end happens.

This means batching never changes which slices get corrected, only how many
forward passes are needed.  The result is numerically identical to sequential
mode when no rollback occurs; with rollback it is equivalent to the original
sequential behaviour (rollback ≤ batch_size − 1 slices are re-predicted).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from inference.ssf import SSFController, BaseSSFStrategy, RelativeSSIMStrategy

# ---------------------------------------------------------------------------
# Project-root path injection
# ---------------------------------------------------------------------------
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.preprocessing import universal_normalization, min_max_norm, shaping
from inference.tiling import TiledInference
from utils.metrics import dice_score_tf


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Structured result returned by VolumeInference.run() and
    InteractiveFeedbackLoop.run().

    Attributes
    ----------
    results_3d : np.ndarray, shape (S, H, W)
        Binary predictions for each evaluated slice, in ascending volume-index
        order: [backward_slices_ascending..., prompt_slice, forward_slices...].
    gt_3d : np.ndarray, shape (S, H, W)
        Corresponding ground-truth binary slices in the same order.
    backward_indices : list[int]
        Original volume slice indices traversed *before* the prompt (descending
        from prompt, but stored here in ascending order for convenience).
    forward_indices : list[int]
        Original volume slice indices traversed *after* the prompt (ascending).
    prompt_axis : int
    prompt_idx : int
    normalization_mode : str
        'universal' or 'legacy' — whichever was actually applied.

    # Metadata for reproducibility
    # ----------------------------
    ssf_strategy : str or None
        Name of the SSF strategy used (e.g. 'RelativeSSIM(t=0.40)').
    gt_dice_threshold : float or None
        IFL threshold used (only for IFL modes).
    """
    results_3d: np.ndarray
    gt_3d: np.ndarray
    backward_indices: List[int]
    forward_indices: List[int]
    prompt_axis: int
    prompt_idx: int
    normalization_mode: str
    # Metadata
    ssf_strategy: Optional[str] = None
    gt_dice_threshold: Optional[float] = None
    # IFL-only — None / empty in plain VolumeInference
    num_user_interacts: Optional[int] = None
    user_interacts_idx: List[int] = field(default_factory=list)
    # Per-slice mean sigmoid confidence (foreground region); always populated
    confidence_per_slice: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Utility: SSIM helper lives in inference.ssf
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Utility: valid slice extraction
# ---------------------------------------------------------------------------

def _extract_valid_slices(
    seg_3d: np.ndarray,
    prompt_axis: int,
    prompt_idx: int,
) -> Tuple[List[int], List[int]]:
    """
    Return (backward_indices, forward_indices) along *prompt_axis*, restricted
    to slices that contain at least one foreground voxel.

    Backward indices run from prompt_idx-1 down to 0 (the loop moves away from
    the prompt).  Forward indices run from prompt_idx+1 upward.
    """
    nonempty = {
        i for i in range(seg_3d.shape[prompt_axis])
        if np.any(np.take(seg_3d, i, axis=prompt_axis))
    }
    backward = [i for i in range(prompt_idx - 1, -1, -1) if i in nonempty]
    forward  = [i for i in range(prompt_idx + 1, seg_3d.shape[prompt_axis]) if i in nonempty]
    return backward, forward


# ---------------------------------------------------------------------------
# Normalization selection
# ---------------------------------------------------------------------------

_LEGACY_BOUNDARY = 292  # first version to use volume-based z-score


def _resolve_normalization(model_path: str | Path, normalization: str) -> str:
    """Determine which normalization mode to apply.  Returns 'universal' or 'legacy'."""
    if normalization in ("universal", "legacy"):
        return normalization
    if normalization != "auto":
        raise ValueError(
            f"normalization must be 'auto', 'universal', or 'legacy'; got '{normalization}'"
        )
    stem  = Path(model_path).stem
    parts = stem.replace("-", "_").split("_")
    for part in reversed(parts):
        try:
            version = int(part)
            return "universal" if version >= _LEGACY_BOUNDARY else "legacy"
        except ValueError:
            continue
    print(
        f"[VolumeInference] WARNING: could not extract version number from "
        f"'{Path(model_path).name}'. Defaulting to 'universal' normalization."
    )
    return "universal"


def _normalize_slice_legacy(slice_2d: np.ndarray) -> np.ndarray:
    """Slice-wise robust min-max (pre-v292).  Returns float32 (H, W)."""
    t = tf.constant(slice_2d[..., np.newaxis], dtype=tf.float32)
    return min_max_norm(t).numpy()[..., 0]


# ---------------------------------------------------------------------------
# VolumeInference — SSF only  (no GT required)
# ---------------------------------------------------------------------------

class VolumeInference:
    """
    Propagate a Prompt-UNet prediction across a 3-D volume using
    Self-Supervised Feedback (SSF).

    The model is loaded and JIT-compiled **once** at construction time.

    Input resolution
    ----------------
    Volumes can be **any spatial size**.  Each 2-D slice is processed at its
    native resolution using adaptive tiling (``inference/tiling.py``)::

        • Slice already 128 × 128  →  single tile, zero overhead.
        • Slice smaller than 128   →  single tile, border pixels repeated (clamp).
        • Slice larger than 128    →  multiple tiles covering the prompt bbox,
                                     predictions blended with tent weights.

    Callers should **not** pre-resize slices.  Prompt masks must be at the
    same native resolution as their corresponding image slices.

    Parameters
    ----------
    model_path : str or Path
        Path to a saved ``.keras`` file.
    normalization : {'auto', 'universal', 'legacy'}
        'auto'      — infer from the version number embedded in the filename.
        'universal' — volume-based z-score (v292+).
        'legacy'    — slice-wise min-max  (pre-v292).
    modality : str
        'CT' or 'MRI'.  Used only when normalization resolves to 'universal'.
    output_threshold : float
        Sigmoid threshold applied to model output to produce binary masks.
    buffer_size : int
        Number of recent predictions kept in the SSF rolling buffer.
        Default 6.
    batch_size : int
        Number of slices predicted per GPU forward pass.  Slices within a
        batch all use the same prompt; if SSF or IFL fires mid-batch, slices
        after the trigger are rolled back and re-predicted with the updated
        prompt.  Default 3.
    tile_trigger_fraction : float
        Passed to ``TiledInference``.  If the prompt bbox on a given axis
        exceeds ``128 * tile_trigger_fraction`` pixels, tiling is applied
        on that axis.  Default 0.75.
    """

    def __init__(
        self,
        model_path: str | Path,
        modality: Optional[str],
        normalization: str = "auto",
        output_threshold: float = 0.5,
        ssf_strategy: Optional[BaseSSFStrategy] = None,
        buffer_size: int = 4,
        batch_size: int = 3,
        tile_trigger_fraction: float = 0.75,
    ):
        self.model_path         = Path(model_path)
        self.normalization_mode = _resolve_normalization(model_path, normalization)
        self.modality           = modality
        self.output_threshold   = output_threshold
        self.buffer_size        = buffer_size
        self.batch_size         = max(1, batch_size)
        self._ssf               = SSFController(ssf_strategy, buffer_size=buffer_size)

        print(
            f"[VolumeInference] Loading '{self.model_path.name}' "
            f"(norm='{self.normalization_mode}', modality_fallback={self.modality}, "
            f"batch_size={self.batch_size})"
        )
        self.model = tf.keras.models.load_model(str(self.model_path))

        # JIT-compile the forward pass once.
        # The batch dimension is dynamic (None) so any number of tiles per slice
        # is handled without re-tracing.
        self._fast_batch_fn = tf.function(
            func=lambda x, p: self.model([x, p], training=False),
            input_signature=[
                tf.TensorSpec([None, 128, 128, 1], dtype=tf.float32),
                tf.TensorSpec([None, 128, 128, 2], dtype=tf.float32),
            ],
        )
        # Warm up so the first real call doesn't pay tracing cost
        _dummy_x = tf.zeros([1, 128, 128, 1])
        _dummy_p = tf.zeros([1, 128, 128, 2])
        self._fast_batch_fn(_dummy_x, _dummy_p)
        print("[VolumeInference] Graph compiled (warm-up done).")

        # Tiler — reused across all slices (stateless weight computation)
        self._tiler = TiledInference(
            tile_size=128,
            tile_trigger_fraction=tile_trigger_fraction,
        )

    def set_ssf_strategy(self, strategy: Optional[BaseSSFStrategy]) -> None:
        """
        Swap the SSF strategy on a live model without reloading weights.

        Useful in tuning loops where you want to test multiple strategies on
        the same loaded model.  The new strategy takes effect on the next
        call to :meth:`run`.

        Parameters
        ----------
        strategy : BaseSSFStrategy or None
            New strategy to use.  ``None`` disables SSF.
        """
        self._ssf.strategy = strategy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_volume(self, img_3d: np.ndarray, modality: Optional[str] = None) -> np.ndarray:
        """Apply appropriate normalization to the entire volume (float32)."""
        mod = modality if modality is not None else self.modality
        if self.normalization_mode == "universal":
            return universal_normalization(img_3d, modality=mod)
        return img_3d.astype(np.float32)  # legacy: per-slice norm applied later

    def _extract_plane(
        self,
        img_vol_normalized: np.ndarray,
        idx: int,
        axis: int,
    ) -> np.ndarray:
        """
        Extract one 2-D slice at native resolution.

        Returns
        -------
        np.ndarray  (H, W) float32 — NOT resized.  Use ``_predict_tiled``
        to run inference on this plane.
        """
        s = np.take(img_vol_normalized, idx, axis=axis).astype(np.float32)  # (H, W)
        if self.normalization_mode == "legacy":
            s = _normalize_slice_legacy(s)
        return s

    def _predict_tiled(
        self,
        image_plane: np.ndarray,
        prompt_img_plane: np.ndarray,
        prompt_mask_plane: np.ndarray,
    ) -> np.ndarray:
        """
        Run tiled inference on a single native-resolution slice.

        Parameters
        ----------
        image_plane      : (H, W) float32 — query image (current slice).
        prompt_img_plane : (H, W) float32 — prompt image channel.
        prompt_mask_plane: (H, W) float32 — prompt mask channel {0, 1}.

        Returns
        -------
        np.ndarray  (H, W) float32 — binary {0, 1} prediction at native resolution.
        """
        def _predict_fn(
            img_batch: np.ndarray,    # (B, 128, 128, 1)
            prompt_batch: np.ndarray, # (B, 128, 128, 2)
        ) -> np.ndarray:              # (B, 128, 128, 1)
            x = tf.constant(img_batch,    dtype=tf.float32)
            p = tf.constant(prompt_batch, dtype=tf.float32)
            return self._fast_batch_fn(x, p).numpy()

        return self._tiler.run(
            image_plane=image_plane,
            prompt_img_plane=prompt_img_plane,
            prompt_mask_plane=prompt_mask_plane,
            predict_fn=_predict_fn,
            threshold=self.output_threshold,
        )


    # ------------------------------------------------------------------
    # IFL hook — overridden by InteractiveFeedbackLoop
    # ------------------------------------------------------------------

    def _maybe_ifl_update(
        self,
        vol_i: int,
        current_slice: tf.Tensor,
        pred: tf.Tensor,
        y_gt: tf.Tensor,
        p: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, bool]:
        """
        Hook called after prediction but before SSF check for each slice.

        Returns
        -------
        (final_pred, new_p, rollback_next)
            final_pred    : the prediction to store (pred unchanged for SSF-only).
            new_p         : the (possibly updated) prompt for subsequent slices.
            rollback_next : True if slices after this one need to be re-predicted.
        """
        return pred, p, False   # base class: no-op

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        img_3d: np.ndarray,
        seg_3d_binary: np.ndarray,
        initial_prompt_2d_seg: np.ndarray,
        prompt_axis: int,
        prompt_idx: int,
        modality: Optional[str] = None,
    ) -> RunResult:
        """
        Propagate the segmentation prompt across all non-empty slices using SSF,
        optionally with IFL (when called from InteractiveFeedbackLoop).

        Parameters
        ----------
        img_3d : np.ndarray, shape (X, Y, Z)
            Raw intensity volume.  Normalization is applied internally.
            Any spatial size is accepted — slices are NOT pre-resized.
        seg_3d_binary : np.ndarray, shape (X, Y, Z)
            Binary ground-truth volume.  In SSF-only mode this is used solely
            to determine which slices to visit and to build gt_3d.
        initial_prompt_2d_seg : np.ndarray, shape (H, W)
            Binary 2-D prompt mask at the prompt slice.  Must be at the
            **native slice resolution** (same H, W as the corresponding image
            slice).  Do NOT pre-resize to 128×128.
        prompt_axis : int   — axis (0, 1, or 2) for slice extraction.
        prompt_idx  : int   — slice index of the prompt.
        modality    : str or None — per-call override (CT/MRI).

        Returns
        -------
        RunResult
            ``results_3d`` has shape ``(S, H_slice, W_slice)`` — native resolution.
        """
        img_3d        = np.squeeze(np.asarray(img_3d, dtype=np.float32))
        seg_3d_binary = np.asarray(seg_3d_binary, dtype=np.float32)

        img_vol_norm = self._normalize_volume(img_3d, modality)

        # ---- Prompt initialisation (native resolution) ----
        # Extract prompt image plane and mask at native resolution
        prompt_img_plane  = self._extract_plane(img_vol_norm, prompt_idx, prompt_axis)  # (H, W)
        prompt_mask_plane = np.asarray(initial_prompt_2d_seg, dtype=np.float32)          # (H, W)

        # The 128×128 tiled prompt tensors are carried forward for SSF/IFL checks.
        # We store the *downsampled* 128×128 representation only for SSIM comparison
        # and prompt updates.  All actual predictions return native-res masks.
        prompt_img_128   = shaping(np.expand_dims(prompt_img_plane, -1))    # (1,128,128,1)
        prompt_mask_128  = shaping(
            np.expand_dims(prompt_mask_plane, -1).astype(np.float32), binary=True
        )                                                                     # (1,128,128,1)
        p_initial = tf.concat([prompt_img_128, prompt_mask_128], axis=-1)   # (1,128,128,2)
        p         = p_initial
        # Keep track of current prompt planes (native res) for tiling
        cur_prompt_img  = prompt_img_plane
        cur_prompt_mask = prompt_mask_plane

        backward_indices, forward_indices = _extract_valid_slices(
            seg_3d_binary, prompt_axis, prompt_idx
        )

        # ---------------------------------------------------------------
        # Slice loop with SSF / IFL rollback
        # ---------------------------------------------------------------
        results                  : List[np.ndarray] = []
        ground_truths            : List[np.ndarray] = []
        confidence_per_slice_lst : List[float]       = []

        self._ssf.reset(prompt_mask_128)

        idx_queue: List[int] = list(backward_indices) + list(forward_indices)

        while idx_queue:
            # Determine mini-batch (never crossing backward→forward boundary)
            b = min(self.batch_size, len(idx_queue))
            if backward_indices and backward_indices[-1] in idx_queue[:b]:
                b = idx_queue.index(backward_indices[-1]) + 1

            batch_idxs = idx_queue[:b]

            # Prepare native-res planes for each slice in the batch
            batch_planes: List[np.ndarray] = [
                self._extract_plane(img_vol_norm, i, prompt_axis) for i in batch_idxs
            ]
            batch_gt_planes: List[np.ndarray] = [
                np.take(seg_3d_binary, i, axis=prompt_axis).astype(np.float32)
                for i in batch_idxs
            ]

            rollback_start = b

            # ----------------------------------------------------------------
            # Batch ALL tiles from ALL slices in this window into one GPU pass.
            #
            # Why: the original code batched multiple slices in one forward
            # pass (batch_size slices × 1 tile).  With tiling, each slice may
            # have N tiles.  We combine all slices × all tiles into a single
            # _fast_batch_fn call so GPU utilisation is maximised.
            #
            # All slices in a batch share the same prompt (cur_prompt_img /
            # cur_prompt_mask), so the tile plan is identical for every slice.
            # ----------------------------------------------------------------
            tile_starts = self._tiler._plan_tiles(
                cur_prompt_mask, *batch_planes[0].shape
            )  # list of (y0, x0) — same for all slices since prompt is shared
            n_tiles = len(tile_starts)

            # Build mega-batch: (b * n_tiles, 128, 128, 1/2)
            all_img_patches    = []  # b × n_tiles entries of (128, 128)
            all_prompt_patches = []  # b × n_tiles entries of (128, 128, 2)
            H_slice, W_slice = batch_planes[0].shape
            for img_plane_k in batch_planes:
                for y0, x0 in tile_starts:
                    img_p  = TiledInference._extract_patch(img_plane_k,  y0, x0, H_slice, W_slice)
                    pimg_p = TiledInference._extract_patch(cur_prompt_img,  y0, x0, H_slice, W_slice)
                    pmsk_p = TiledInference._extract_patch(cur_prompt_mask, y0, x0, H_slice, W_slice)
                    all_img_patches.append(img_p)
                    all_prompt_patches.append(np.stack([pimg_p, pmsk_p], axis=-1))

            img_mega    = np.stack(all_img_patches,    axis=0)[:, :, :, np.newaxis]  # (b*t,128,128,1)
            prompt_mega = np.stack(all_prompt_patches, axis=0)                        # (b*t,128,128,2)
            pred_mega   = self._fast_batch_fn(
                tf.constant(img_mega,    dtype=tf.float32),
                tf.constant(prompt_mega, dtype=tf.float32),
            ).numpy()   # (b*t, 128, 128, 1)

            # Blend per-slice predictions from the mega-batch; also keep raw
            # sigmoid probabilities for ConfidenceDropStrategy.
            batch_preds     : List[np.ndarray] = []  # binary
            batch_probs_raw : List[np.ndarray] = []  # float32 sigmoid [0,1]
            for k in range(b):
                prob_slice = pred_mega[k * n_tiles:(k + 1) * n_tiles]  # (n_tiles,128,128,1)
                accum_prob   = np.zeros((H_slice, W_slice), dtype=np.float32)
                accum_weight = np.zeros((H_slice, W_slice), dtype=np.float32)
                T = self._tiler.tile_size
                w2d = self._tiler._weight2d
                for ti, (y0, x0) in enumerate(tile_starts):
                    prob_patch = prob_slice[ti, :, :, 0]
                    y_end = min(y0 + T, H_slice)
                    x_end = min(x0 + T, W_slice)
                    oy, ox = y_end - y0, x_end - x0
                    accum_prob  [y0:y_end, x0:x_end] += prob_patch[:oy, :ox] * w2d[:oy, :ox]
                    accum_weight[y0:y_end, x0:x_end] += w2d[:oy, :ox]
                eps = 1e-8
                prob = np.where(accum_weight > eps,
                                accum_prob / (accum_weight + eps), 0.0)
                batch_probs_raw.append(prob)   # raw float32 sigmoid
                batch_preds.append((prob >= self.output_threshold).astype(np.float32))

            # --- Process slice-by-slice for SSF / IFL checks ---
            for k in range(b):
                vol_i       = batch_idxs[k]
                img_plane_k = batch_planes[k]      # (H, W)
                gt_plane_k  = batch_gt_planes[k]   # (H, W)

                # Pre-computed native-resolution prediction from mega-batch
                pred_native = batch_preds[k]        # (H, W) binary float32
                prob_raw_k  = batch_probs_raw[k]    # (H, W) float32 sigmoid

                # 128×128 thumbnails for SSF checks
                pred_128     = shaping(np.expand_dims(pred_native, -1), binary=True)  # (1,128,128,1)
                pred_prob_128 = shaping(np.expand_dims(prob_raw_k, -1))               # (1,128,128,1)

                # GT at 128×128 for IFL Dice check
                gt_128 = shaping(
                    np.expand_dims(gt_plane_k, -1), binary=True
                )   # (1, 128, 128, 1)

                # 1. IFL hook — base class is no-op; IFL subclass may substitute GT
                final_pred_128, p, ifl_rollback = self._maybe_ifl_update(
                    vol_i, prompt_img_128, pred_128, gt_128, p
                )

                # If IFL fired, update cur_prompt_mask from the accepted GT
                if ifl_rollback:
                    cur_prompt_mask = gt_plane_k.copy()
                    cur_prompt_img  = img_plane_k.copy()
                    prompt_img_128  = shaping(np.expand_dims(cur_prompt_img, -1))
                else:
                    cur_prompt_mask = pred_native.copy()

                # Track per-slice confidence (mean sigmoid over predicted foreground)
                fg = pred_native > 0.5
                confidence_per_slice_lst.append(
                    float(prob_raw_k[fg].mean()) if fg.any() else 0.0
                )

                # Store native-resolution results
                ground_truths.append(gt_plane_k)
                results.append(pred_native)

                if ifl_rollback:
                    self._ssf.reset_trigger()  # reset strategy ref without clearing buffer
                    rollback_start = k + 1
                    break

                # 2. SSF check — delegated to SSFController
                img_plane_128 = shaping(np.expand_dims(img_plane_k, -1))  # (1,128,128,1)
                ssf_fired, new_mask_128 = self._ssf.update(
                    slice_idx       = vol_i,
                    img_plane_128   = img_plane_128[0, ..., 0].numpy(),
                    pred_binary_128 = pred_128,
                    pred_prob_128   = pred_prob_128,
                    prompt_img_128  = p[0, ..., 0].numpy(),
                )

                if ssf_fired:
                    # new_mask_128 is (1,128,128,1) binary TF tensor
                    p               = tf.concat([img_plane_128, new_mask_128], axis=-1)
                    cur_prompt_mask = shaping(
                        np.expand_dims(new_mask_128[0, ..., 0].numpy(), -1), binary=True,
                        h=img_plane_k.shape[0], w=img_plane_k.shape[1]
                    )[0, :, :, 0].numpy()
                    cur_prompt_img  = img_plane_k.copy()
                    rollback_start  = k + 1
                    break

            idx_queue = batch_idxs[rollback_start:] + idx_queue[b:]

            if backward_indices and backward_indices[-1] in batch_idxs[:rollback_start]:
                results                  = list(reversed(results))
                ground_truths            = list(reversed(ground_truths))
                confidence_per_slice_lst = list(reversed(confidence_per_slice_lst))

                # Insert prompt slice itself in the middle
                results.append(prompt_mask_plane.copy())
                ground_truths.append(prompt_mask_plane.copy())
                confidence_per_slice_lst.append(0.0)  # prompt slice has no model confidence

                # Reset for forward pass
                p               = p_initial
                cur_prompt_img  = prompt_img_plane
                cur_prompt_mask = prompt_mask_plane
                self._ssf.reset(prompt_mask_128)

        if not backward_indices:
            results.append(prompt_mask_plane.copy())
            ground_truths.append(prompt_mask_plane.copy())
            confidence_per_slice_lst.append(0.0)

        results_arr = np.stack(results, axis=0)        # (S, H, W)
        gt_arr      = np.stack(ground_truths, axis=0)  # (S, H, W)

        return RunResult(
            results_3d           = results_arr,
            gt_3d                = gt_arr,
            backward_indices     = backward_indices,
            forward_indices      = forward_indices,
            prompt_axis          = prompt_axis,
            prompt_idx           = prompt_idx,
            normalization_mode   = self.normalization_mode,
            ssf_strategy         = self._ssf.strategy.name if self._ssf.strategy else None,
            confidence_per_slice = confidence_per_slice_lst,
        )


# ---------------------------------------------------------------------------
# InteractiveFeedbackLoop — SSF + IFL  (GT required)
# ---------------------------------------------------------------------------

class InteractiveFeedbackLoop(VolumeInference):
    """
    Extends VolumeInference with an Interactive Feedback Loop (IFL).

    When the model's Dice on a slice drops below `gt_dice_threshold`, the
    ground-truth mask is substituted for the prediction and used as the next
    prompt.  Slices after the corrected one are rolled back and re-predicted
    with the updated prompt — so batching never silently uses a stale prompt.

    The corrected slice indices (`user_interacts_idx`) are returned in
    RunResult and should be forwarded to NNInteractiveInference so that
    nnInteractive receives the same interaction budget.

    Parameters
    ----------
    gt_dice_threshold : float
        Dice below this value triggers a GT correction.  Default 0.65.
    (All other parameters are inherited from VolumeInference.)
    """

    def __init__(self, *args, gt_dice_threshold: float = 0.65, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_dice_threshold   = gt_dice_threshold
        self._ifl_user_interacts : List[int] = []
        self._ifl_enabled        : bool = True   # can be toggled without reload

    def set_ifl_enabled(self, enabled: bool) -> None:
        """
        Enable or disable Interactive Feedback Loop correction at runtime.

        When ``enabled=False`` the instance behaves identically to plain
        :class:`VolumeInference`, applying only SSF (if a strategy is set).
        This lets one loaded model be benchmarked under multiple mode
        configurations without reloading weights.

        Parameters
        ----------
        enabled : bool
            ``True``  — IFL correction active (default).
            ``False`` — IFL correction disabled; GT substitution never fires.
        """
        self._ifl_enabled = enabled

    # Override the IFL hook — called per-slice inside VolumeInference.run()
    def _maybe_ifl_update(
        self,
        vol_i: int,
        current_slice: tf.Tensor,
        pred: tf.Tensor,
        y_gt: tf.Tensor,
        p: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, bool]:
        """
        Check Dice of prediction against GT.  If below threshold:
          • Replace prediction with GT (user corrects the slice).
          • Update prompt p for subsequent slices.
          • Signal rollback of any further slices in the current batch
            (they were predicted with the old, now-invalidated prompt).

        When IFL is disabled (set_ifl_enabled(False)) this is a no-op,
        matching the base VolumeInference behaviour.
        """
        if not self._ifl_enabled:
            return pred, p, False   # disabled: no GT substitution
        current_dice = float(dice_score_tf(y_gt, pred).numpy())
        if current_dice < self.gt_dice_threshold:
            self._ifl_user_interacts.append(vol_i)
            new_p = tf.concat([current_slice, y_gt], axis=-1)
            return y_gt, new_p, True   # GT substitution + rollback next
        return pred, p, False

    def run(
        self,
        img_3d: np.ndarray,
        seg_3d_binary: np.ndarray,
        initial_prompt_2d_seg: np.ndarray,
        prompt_axis: int,
        prompt_idx: int,
        modality: Optional[str] = None,
    ) -> RunResult:
        """
        Same signature as VolumeInference.run().  Internally activates the IFL
        hook and enriches the returned RunResult with interaction metadata.

        Parameters
        ----------
        modality : str or None — per-call modality override.

        Returns
        -------
        RunResult with `num_user_interacts` and `user_interacts_idx` populated.
        """
        # Reset interaction log for this run
        self._ifl_user_interacts = []

        # Delegate to the shared mini-batch loop (which calls _maybe_ifl_update)
        result = super().run(
            img_3d, seg_3d_binary, initial_prompt_2d_seg,
            prompt_axis, prompt_idx, modality=modality,
        )

        # Patch IFL-specific fields onto the result.
        # When IFL is disabled, report None / [] so callers can distinguish
        # SSF-only runs from IFL runs.
        if self._ifl_enabled:
            result.num_user_interacts = len(self._ifl_user_interacts) + 1  # +1 for initial
            result.user_interacts_idx = list(self._ifl_user_interacts)
            result.gt_dice_threshold  = self.gt_dice_threshold
        else:
            result.num_user_interacts = None
            result.user_interacts_idx = []
            result.gt_dice_threshold  = None
        return result


# ---------------------------------------------------------------------------
# Convenience: generate an initial prompt from a labelled 3-D volume
# ---------------------------------------------------------------------------

def generate_initial_prompt(
    seg_3d: np.ndarray,
    min_pixels: int = 50,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], int]:
    """
    Select a random ROI from a multi-label 3-D segmentation and generate a
    2-D prompt mask on a randomly chosen slice.

    Parameters
    ----------
    seg_3d      : np.ndarray, integer labels (0 = background).
    min_pixels  : int — minimum foreground pixels for an ROI to be eligible.
    visualize   : bool — show the selected prompt mask (requires matplotlib).

    Returns
    -------
    initial_prompt_3d   : np.ndarray, shape == seg_3d.shape, float32.
        Zeros everywhere except the selected 2-D slice (binary ROI mask).
    initial_prompt_2d   : np.ndarray, shape (H, W), float32.
    (axis, slice_idx)   : Tuple[int, int].
    selected_roi        : int — the chosen label value.
    """
    seg_3d = np.asarray(seg_3d)

    def _nonempty_slices(arr, axis):
        return [i for i in range(arr.shape[axis]) if np.any(np.take(arr, i, axis=axis))]

    axis       = random.choice([0, 1, 2])
    valid_idxs = _nonempty_slices(seg_3d, axis)
    if not valid_idxs:
        raise ValueError("seg_3d has no non-empty slices.")

    for _attempt in range(50):
        rand_i     = random.choice(valid_idxs)
        rand_slice = np.take(seg_3d, rand_i, axis=axis)
        labels     = np.unique(rand_slice)
        labels     = labels[labels != 0]
        eligible   = [lbl for lbl in labels if np.sum(rand_slice == lbl) >= min_pixels]
        if eligible:
            selected_roi = random.choice(eligible)
            break
    else:
        raise ValueError("Could not find an eligible ROI after 50 attempts.")

    roi_mask          = (rand_slice == selected_roi).astype(np.float32)
    initial_prompt_3d = np.zeros_like(seg_3d, dtype=np.float32)
    if axis == 0:
        initial_prompt_3d[rand_i, :, :]  = roi_mask
    elif axis == 1:
        initial_prompt_3d[:, rand_i, :]  = roi_mask
    else:
        initial_prompt_3d[:, :, rand_i]  = roi_mask

    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(roi_mask)
        plt.title(
            f"ROI {selected_roi} | Axis {axis}, Slice {rand_i} | "
            f"{int(np.sum(roi_mask))} px"
        )
        plt.show()

    return initial_prompt_3d, roi_mask, (axis, rand_i), selected_roi
