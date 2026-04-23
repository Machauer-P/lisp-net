# Adaptive Tiling

## How it works

The adaptive tiling mechanism allows Prompt-UNet to process images and volumes of any native resolution without resizing, maintaining high-fidelity predictions through seamless tile blending.

- **Adaptive BBox Selection**: The logic calculates the bounding box of the prompt mask and only creates tiles along axes where the structure exceeds 96 pixels (75% of the 128x128 model input size).
- **Tent-Weight Blending**: Employs a tent-weighting function (`w = 0.1 + 0.9*(1-|t|)` squared logic) to seamlessly merge overlapping patches and prevent seam artifacts at tile boundaries.
- **Mega-Batching**: Collects all tiles from all slices within a `batch_size` window and evaluates them together in a single GPU forward pass for high throughput.
- **Vectorized Extraction**: Utilizes NumPy fancy indexing (`np.ix_()`) for high-speed, framework-agnostic patch extraction.
- **Pre-computed Weights**: Seamless blending weight maps are computed once at instantiation rather than dynamically per slice or frame.

## What changed from the original KSegmentation.js

The Python port is conceptually identical to the JS original but introduces several performance optimizations specifically tailored for backend GPU execution:

| Feature | KSegmentation.js (Original) | Python Implementation (Improved) |
| :--- | :--- | :--- |
| **Throughput** | Single-tile processing (Serial `model.predict`) | **Mega-batching** (Cross-slice parallel) |
| **Memory / Setup** | Recalculated weights per frame | **Pre-computed** stateless weight maps |
| **Patch Extraction**| JS pixel-loop interpolation | **NumPy vectorized** gather operations |
| **Coupling** | Implicitly tied to tfjs UI state | Framework-agnostic `predict_fn` API |

> [IMPORTANT]
> **Do NOT pre-resize inputs to 128×128.**
> Pass the native slice/image and let the tiler handle it. Best to use inference.py or volume_inference.py which already include the tiling module.