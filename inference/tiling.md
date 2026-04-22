# Walkthrough: Robust Inference with Adaptive Tiling

This walkthrough documents the full transition from naive resizing to an
adaptive tiling pipeline across both 2D and 3D inference paths in Prompt-UNet.

---

## 1. Core Tiling Module (`inference/tiling.py`)

Ported the adaptive tiling algorithm from `KSegmentation.js` and optimized it for Python/NumPy.

- **Adaptive BBox Selection**: Tiles only axes where the prompt structure exceeds 96 px (75 % of 128).
- **Tent-Weight Blending**: Exact match to the JS weighting `w = 0.1 + 0.9*(1-|t|)`, preventing seam artifacts at tile boundaries.
- **Vectorized Extraction**: NumPy `np.ix_()` replaces JS pixel-by-pixel loops.
- **Weight Map Pre-Computed**: Done once at `__init__`, not per slice/frame.

---

## 2. 3D Volume Inference (`inference/volume_inference.py`)

Rewired `VolumeInference` to produce native-resolution predictions:

- `_prepare_slice()` + `_predict_batch()` replaced by `_extract_plane()` + mega-batch tiling.
- **Mega-Batching**: All tiles across all slices in a `batch_size` window are fused into a single `_fast_batch_fn` call.
- `results_3d` now stores `(S, H_slice, W_slice)` arrays at native resolution.
- SSF / IFL trigger checks still use 128×128 thumbnails (fast, no quality impact on the control flow).

---

## 3. 3D Benchmark Reconstruction (`run_3d_benchmark.py`)

- `_reconstruct_volume()` simplified to a direct `pred_vol[vol_idx] = s` assignment.
- **No resize anywhere** — eliminated the `tf.image.resize` hack that caused broadcast errors.

---

## 4. 2D Inference Wrapper (`inference/p_unet_inference.py`)

Added tiling support while keeping the 2D evaluation pipeline **100 % unchanged**.

| Input spatial size | Path | Behaviour |
|---|---|---|
| Exactly 128 × 128 | **Fast path** | Single batched `_fast_predict_fn` call. No overhead. *(Used by `eval_pipeline_2d.py`)* |
| < 128 × 128 | **Tiling path** | Single patch, border pixels edge-clamped. Output at native resolution. |
| > 128 × 128 | **Tiling path** | Adaptive bbox-guided tiles, tent-weight blended. Output at native resolution. |

The class-level docstring also makes the three input conventions explicit:
1. Pre-cropped 128 × 128 (fastest)
2. Native resolution (recommended for production use)
3. Explicit resize before calling `predict()` (acceptable for isotropic regions only)

**Warning included**: do NOT blindly resize non-square regions (e.g. 231 × 270 → 128 × 128); use native input instead.

---

## 5. Summary: JS Original vs. Python Implementation

| Feature | KSegmentation.js | Python |
|---|---|---|
| Throughput | Serial tile-per-tile (`model.predict` ×N) | **Mega-batched** (all slices × tiles in one pass) |
| Patch extraction | JS pixel loop | **Vectorized** `np.ix_()` gather |
| Weight map | Recomputed every frame | **Pre-computed** at construction |
| Framework coupling | tfjs in browser | Framework-agnostic `predict_fn` API |

---

> [!IMPORTANT]
> **Callers should NOT pre-resize inputs to 128×128.**
> For 2D evaluation, inputs are already 128×128 (fast path, zero overhead).
> For any other use, pass the native slice/image and let the tiler handle it.


We have successfully resolved the broadcasting errors in the 3D volume reconstruction pipeline by implementing a robust, adaptive tiling mechanism. This replaces the previous "naive resize" approach which caused spatial distortion and mathematical mismatches during volume assembly.

## Accomplishments

### 1. Robust Tiling Module (`inference/tiling.py`)
We ported the adaptive tiling algorithm from `KSegmentation.js` and optimized it for Python/NumPy performance.

- **Adaptive BBox Selection**: The logic calculates the bounding box of the prompt mask and only tiles axes where the structure exceeds 96px (75% of 128px).
- **Tent-Weight Blending**: Implemented the exact weighting function from JS (`w = (0.1 + 0.9*(1-|t|))^2`) to ensure seamless transitions in overlapping regions.
- **Vectorized Extraction**: Replaced JS-style pixel loops with NumPy fancy indexing (`np.ix_`) for high-speed patch extraction.

### 2. High-Performance Batching (`inference/volume_inference.py`)
We extended the original JS logic to take full advantage of GPU parallelism.

- **Mega-Batching**: While JS processes one tile at a time, our implementation collects **all tiles from all slices** within a `batch_size` window and executes them in a **single GPU forward pass**.
- **Rollback Compatibility**: The batching logic remains compatible with SSF (Self-Supervised Feedback) and IFL (Interactive Feedback Loop) triggers. If a prompt update is required mid-batch, the system correctly rolls back and re-predicts.

### 3. Native Resolution Reconstruction (`run_3d_benchmark.py`)
The benchmarking pipeline no longer relies on lossy resizing.

- **Pixel-Exact Mapping**: Predictions are generated at the native resolution of the input slice.
- **Direct Assignment**: `_reconstruct_volume` now assigns 2D results directly into the 3D volume grid without intermediate `tf.image.resize` calls.

---

## Technical Comparison: Python Port vs. JS Original

| Feature | KSegmentation.js (Original) | `inference/tiling.py` (Improved) |
| :--- | :--- | :--- |
| **Throughput** | Single-tile processing (Serial) | **Mega-batching** (Cross-slice parallel) |
| **Memory** | Recalculated weights per frame | **Pre-computed** stateless weight maps |
| **Logic** | Implicitly tied to UI state | Framework-agnostic `predict_fn` API |
| **Performance** | JS pixel-loop interpolation | **NumPy vectorized** gather operations |

---

> [!IMPORTANT]
> **Input Constraint Update**: 
> You (the user) should now input volumes of **any size**. Do not pre-resize to 128x128. If the input is already 128x128, the system automatically bypasses the tiling overhead (Single-Tile mode).