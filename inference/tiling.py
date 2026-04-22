"""
inference/tiling.py
===================
Adaptive tiling for Prompt-UNet inference on arbitrary-resolution 2-D slices.

Algorithm (matches KSegmentation.js ``testTFJS``, lines 1556-1941)
-------------------------------------------------------------------
1.  Compute the bounding box of the prompt mask on the current 2-D plane.
2.  Decide per-axis tiling:
      • bbox extent > tile_size * tile_trigger_fraction  →  tile that axis.
      • smaller bbox (or no bbox)                        →  single centered tile.
3.  ``compute_adaptive_starts``  — minimum-count tile coverage of the bbox.
4.  Extract ``(tile_size, tile_size)`` patches (with edge-clamping for OOB).
5.  Run the model on all patches (batched single forward pass).
6.  Tent-weight blend: ``accumProb += pred * w``, ``accumWeight += w``.
7.  Threshold: ``accumProb / accumWeight > threshold``.

Usage
-----
Create a :class:`TiledInference` instance (once, reuse across slices)::

    tiler = TiledInference(tile_size=128)

    # predict_fn: callable (img_batch, prompt_batch) -> prob_batch
    #   img_batch    : np.ndarray  (B, 128, 128, 1)   float32, normalized image
    #   prompt_batch : np.ndarray  (B, 128, 128, 2)   float32, [img_channel, mask_channel]
    #   returns      : np.ndarray  (B, 128, 128, 1)   float32  sigmoid probabilities

    binary_mask = tiler.run(
        image_plane     = img_slice_2d,       # (H, W) float32, normalized
        prompt_img_plane= prompt_img_slice,   # (H, W) float32, prompt image channel
        prompt_mask_plane= prompt_mask_slice, # (H, W) float32, binary {0,1}
        predict_fn      = my_model_fn,
        threshold       = 0.5,
    )

Input sizes
-----------
* Exact  128 × 128  →  zero overhead, single pass, identity transform.
* Smaller than 128  →  1 tile, edge-clamped (same as JS ``clamp()``).
* Larger than 128   →  multiple tiles; blended result at native resolution.

No resize is ever applied.  Callers should NOT pre-resize slices.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TiledInference:
    """
    Adaptive bbox-guided tiled inference for a fixed-input-size segmentation model.

    Parameters
    ----------
    tile_size : int
        Model input spatial size.  Must be 128 for Prompt-UNet.
    tile_trigger_fraction : float
        If the prompt bbox extent on a given axis exceeds
        ``tile_size * tile_trigger_fraction``, multiple tiles are placed
        along that axis.  Otherwise a single centered tile is used.
        Default 0.75 (= 96 px for tile_size=128).
    """

    def __init__(
        self,
        tile_size: int = 128,
        tile_trigger_fraction: float = 0.75,
    ):
        self.tile_size             = tile_size
        self.tile_trigger_fraction = tile_trigger_fraction
        self._trigger_px           = tile_size * tile_trigger_fraction

        # Pre-compute 1-D tent weights: w[i] = 0.1 + 0.9*(1 - |t|), t in [-1,1]
        # These are the same weights used in KSegmentation.js.
        n = tile_size
        t = (2 * np.arange(n) / max(n - 1, 1)) - 1.0  # shape (n,)
        self._weight1d = (0.1 + 0.9 * (1.0 - np.abs(t))).astype(np.float32)
        # 2-D outer product → (tile_size, tile_size) weight map
        self._weight2d = np.outer(self._weight1d, self._weight1d).astype(np.float32)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        image_plane: np.ndarray,
        prompt_img_plane: np.ndarray,
        prompt_mask_plane: np.ndarray,
        predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Run tiled inference on a single 2-D slice.

        Parameters
        ----------
        image_plane : (H, W) float32
            Normalized image at the current slice (the *query* image channel).
        prompt_img_plane : (H, W) float32
            Normalized image at the *prompt* slice (channel 0 of the prompt).
        prompt_mask_plane : (H, W) float32
            Binary mask at the prompt slice (channel 1 of the prompt).
        predict_fn : callable
            ``predict_fn(img_batch, prompt_batch) -> prob_batch``
            where shapes are ``(B, T, T, 1)``, ``(B, T, T, 2)``, ``(B, T, T, 1)``
            and T = ``tile_size``.
        threshold : float
            Sigmoid probability threshold for binary output.  Default 0.5.

        Returns
        -------
        np.ndarray  (H, W) float32 — binary {0, 1} prediction at native resolution.
        """
        image_plane      = np.asarray(image_plane,      dtype=np.float32)
        prompt_img_plane = np.asarray(prompt_img_plane, dtype=np.float32)
        prompt_mask_plane = np.asarray(prompt_mask_plane, dtype=np.float32)

        H, W = image_plane.shape
        T    = self.tile_size

        # --- 1. Compute tile starts ---
        tile_starts = self._plan_tiles(prompt_mask_plane, H, W)

        # --- 2. Extract all patches ---
        img_patches    = []   # list of (T, T)
        prompt_patches = []   # list of (T, T, 2)

        for y0, x0 in tile_starts:
            img_p  = self._extract_patch(image_plane, y0, x0, H, W)
            pimg_p = self._extract_patch(prompt_img_plane, y0, x0, H, W)
            pmsk_p = self._extract_patch(prompt_mask_plane, y0, x0, H, W)
            img_patches.append(img_p)
            prompt_patches.append(np.stack([pimg_p, pmsk_p], axis=-1))  # (T,T,2)

        # --- 3. Batched forward pass ---
        B = len(tile_starts)
        img_batch    = np.stack(img_patches, axis=0)[:, :, :, np.newaxis]   # (B,T,T,1)
        prompt_batch = np.stack(prompt_patches, axis=0)                      # (B,T,T,2)
        prob_batch   = predict_fn(img_batch, prompt_batch)                   # (B,T,T,1)
        prob_batch   = np.asarray(prob_batch, dtype=np.float32)

        # --- 4. Tent-weight blend ---
        accum_prob   = np.zeros((H, W), dtype=np.float32)
        accum_weight = np.zeros((H, W), dtype=np.float32)

        for i, (y0, x0) in enumerate(tile_starts):
            prob_patch = prob_batch[i, :, :, 0]   # (T, T)
            # Destination region (columns / rows inside the image)
            y_end = min(y0 + T, H)
            x_end = min(x0 + T, W)
            oy = y_end - y0   # valid tile height (T unless clipped at bottom)
            ox = x_end - x0   # valid tile width

            accum_prob  [y0:y_end, x0:x_end] += prob_patch[:oy, :ox] * self._weight2d[:oy, :ox]
            accum_weight[y0:y_end, x0:x_end] += self._weight2d[:oy, :ox]

        # --- 5. Normalise + threshold ---
        eps = 1e-8
        prob = np.where(accum_weight > eps, accum_prob / (accum_weight + eps), 0.0)
        return (prob >= threshold).astype(np.float32)

    # ------------------------------------------------------------------
    # Tile planning
    # ------------------------------------------------------------------

    def _plan_tiles(
        self,
        prompt_mask_plane: np.ndarray,
        H: int,
        W: int,
    ) -> List[Tuple[int, int]]:
        """
        Return list of ``(y0, x0)`` tile top-left corners.

        Follows the KSegmentation.js adaptive logic:
        * No visible prompt → full regular coverage.
        * Prompt found     → adaptive minimum-tile coverage per axis.
        """
        T   = self.tile_size
        trig = self._trigger_px
        bbox = _compute_mask_bbox(prompt_mask_plane, threshold=0.5)

        if bbox is None:
            # No prompt visible on this slice → tile the entire plane
            y_starts = _compute_tile_starts(H, T, T)
            x_starts = _compute_tile_starts(W, T, T)
        else:
            tile_y = (bbox["maxY"] - bbox["minY"] + 1) > trig
            tile_x = (bbox["maxX"] - bbox["minX"] + 1) > trig
            y_starts = _compute_adaptive_starts(
                H, T, T,
                bbox["minY"], bbox["maxY"],
                should_tile=tile_y,
            )
            x_starts = _compute_adaptive_starts(
                W, T, T,
                bbox["minX"], bbox["maxX"],
                should_tile=tile_x,
            )

        return [(y, x) for y in y_starts for x in x_starts]

    # ------------------------------------------------------------------
    # Patch extraction (with edge-clamping — no zero padding)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_patch(
        plane: np.ndarray,
        y0: int,
        x0: int,
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Extract a (tile_size, tile_size) patch starting at (y0, x0).

        Pixels that would fall outside the image are filled by repeating the
        nearest edge pixel (``np.clip`` → same as JS ``clamp()``).
        Vectorised: no Python loops.
        """
        T = 128  # fixed for Prompt-UNet
        # Row and column indices into the source plane, clamped to valid range
        rows = np.clip(np.arange(y0, y0 + T), 0, H - 1)   # (T,)
        cols = np.clip(np.arange(x0, x0 + T), 0, W - 1)   # (T,)
        # Outer index → (T, T) patch
        return plane[np.ix_(rows, cols)].astype(np.float32)



# ---------------------------------------------------------------------------
# Helpers (module-level, no class state)
# ---------------------------------------------------------------------------

def _compute_mask_bbox(
    mask: np.ndarray,
    threshold: float = 0.5,
) -> Optional[dict]:
    """Return bbox dict or None if mask is empty."""
    ys, xs = np.where(mask > threshold)
    if len(ys) == 0:
        return None
    return {
        "minY": int(ys.min()), "maxY": int(ys.max()),
        "minX": int(xs.min()), "maxX": int(xs.max()),
    }


def _compute_tile_starts(length: int, tile: int, step: int) -> List[int]:
    """
    Regular grid of tile starts covering [0, length).
    Always includes a tile anchored at ``length - tile`` so the last
    tile covers the end.
    """
    if length <= tile:
        return [0]
    starts: List[int] = []
    s = 0
    while s <= length - tile:
        starts.append(s)
        s += step
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _compute_adaptive_starts(
    length: int,
    tile: int,
    step: int,          # kept for API consistency; not used in adaptive path
    bbox_min: int,
    bbox_max: int,
    should_tile: bool,
) -> List[int]:
    """
    Minimum-tile coverage of [bbox_min, bbox_max].

    Matches ``computeAdaptiveStarts`` in KSegmentation.js exactly:
    * If ``should_tile`` is False → single tile centered on the bbox.
    * Otherwise → minimum number of tiles so [bbox_min, bbox_max] is covered.
    """
    if length <= tile:
        return [0]

    if not should_tile:
        center = 0.5 * (bbox_min + bbox_max)
        start  = int(round(center - tile / 2))
        start  = int(np.clip(start, 0, length - tile))
        return [start]

    length_minus_tile = length - tile
    extent  = bbox_max - bbox_min + 1
    n_tiles = max(1, int(np.ceil(extent / tile)))
    last_needed = int(np.clip(bbox_max - tile + 1, 0, length_minus_tile))

    while True:
        starts = [0] * n_tiles
        starts[n_tiles - 1] = last_needed
        for i in range(n_tiles - 2, -1, -1):
            starts[i] = max(0, starts[i + 1] - tile)
        if starts[0] <= bbox_min:
            break
        n_tiles += 1

    return starts
