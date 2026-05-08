import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Project root on sys.path so inference.tiling is importable regardless of
# where this module is imported from.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from inference.tiling import TiledInference


class PromptUNetPredictor:
    """
    Fast inference wrapper for Prompt U-Net.

    Accepts images and prompts at **any** spatial resolution — no
    pre-resizing required.

    Input modes
    -----------
    1. **Pre-cropped 128 × 128 patches** (batch or single)::

           image  : (..., 128, 128, ...)
           prompt : (..., 128, 128, 2)

       The fastest path — no tiling overhead, single batched forward pass.
       This is what the 2-D evaluation pipeline (``eval_pipeline_2d.py``)
       uses; the behaviour is identical to previous versions.

    2. **Native resolution** (any H, W)::

           image  : (H, W) or (B, H, W) or (H, W, 1) or (B, H, W, 1)
           prompt : (H, W, 2) or (B, H, W, 2)

       ``TiledInference`` is invoked automatically:

       * H ≤ 128 **and** W ≤ 128 → single 128 × 128 patch, border pixels
         repeated (edge-clamping, no zero-padding).
       * H > 128 **or** W > 128  → adaptive bbox-guided tiling with
         tent-weight blending.  The output mask is returned at the same
         (H, W) as the input — **no lossy resize**.

    3. **Square patches resized to 128 × 128** (explicit, before calling
       ``predict``) — acceptable only when the source region is roughly
       isotropic (e.g. 256 × 256 → 128 × 128 with bilinear resize).

    .. warning::

       Do **NOT** blindly resize non-square regions (e.g. 231 × 270 →
       128 × 128). This squashes or stretches anatomy and degrades model
       accuracy. Use native input (mode 2) instead and let the tiler
       handle arbitrary sizes.

    .. danger:: **Normalization Requirement**

       This class does **NOT** normalize raw medical data (e.g. it does NOT
       clip Hounsfield units or apply MRI percentiles). It assumes inputs are
       **already precisely normalized** to the range expected by the loaded
       Prompt-UNet model (usually a z-score clipped to ``[-5, 5]`` for 
       'universal' models >= v292, or ``[0, 1]`` for legacy models). 
       
       Passing raw intensities directly into ``predict()`` will silently
       yield garbage predictions. If you are predicting on raw 3D medical
       volumes, use ``inference_volume.VolumeInference`` instead, which 
       handles normalization automatically via the ``modality`` parameter.

    Parameters
    ----------
    model_path_or_obj : str, Path, or tf.keras.Model
        Path to a saved ``.keras`` model file, or a pre-loaded model.
    """

    def __init__(self, model_path_or_obj):
        if isinstance(model_path_or_obj, (str, Path)):
            self.model = tf.keras.models.load_model(str(model_path_or_obj))
        else:
            self.model = model_path_or_obj

        # JIT-compiled forward pass.
        # input_signature locks to 128×128 so TF never re-traces on batch
        # size changes.  The tiling branch calls this internally for each
        # 128×128 patch, so the same compiled graph is reused in both paths.
        self._fast_predict_fn = tf.function(
            self._fast_signature,
            input_signature=[
                tf.TensorSpec([None, 128, 128, 1], tf.float32),  # image
                tf.TensorSpec([None, 128, 128, 2], tf.float32),  # prompt
            ],
        )

        # Tiler — stateless, reused across all predict() calls.
        # Only activated when the input spatial size is not 128 × 128.
        self._tiler = TiledInference(tile_size=128)

    def _fast_signature(self, x, p):
        """
        Wrapped function for direct tensor execution.

        Bypasses Keras data-pipeline overhead for very large speedups on
        single items / small batches.

        .. important::

           Never falls through to ``model.predict()`` which re-creates a
           ``tf.data`` pipeline on every call — that overhead dominates
           for chunk-by-chunk evaluation loops (~0.5 s wasted per call).
        """
        return self.model([x, p], training=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image, prompt, batch_size=32, threshold=0.5):
        """
        Predict segmentation given image(s) and prompt(s).

        Parameters
        ----------
        image : array-like
            Shape ``(H, W)``, ``(H, W, C)``, ``(B, H, W)``, or
            ``(B, H, W, C)``.  For the tiling path, C must be 1 (or 0
            channel dim).
        prompt : array-like
            Shape matching the batch size, with 2 channels at the last
            axis, e.g. ``(H, W, 2)`` or ``(B, H, W, 2)``.
            Channel 0 = prompt image, channel 1 = prompt binary mask.
        batch_size : int
            Used to chunk batches in the 128 × 128 fast path to avoid
            OOM errors.  Ignored in the tiling path (each sample's tiles
            are batched internally).
        threshold : float
            Sigmoid probability threshold for binary output.

        Returns
        -------
        np.ndarray, float32
            Binary mask.  For 128 × 128 inputs the shape mirrors the
            input (batch / channel dims preserved as in the original).
            For native-resolution inputs the output has shape
            ``(H, W)`` (single sample) or ``(B, H, W)`` (batch).
        """
        x = np.asarray(image,  dtype=np.float32)
        p = np.asarray(prompt, dtype=np.float32)

        original_ndim_x  = x.ndim
        original_shape_x = x.shape

        # ----------------------------------------------------------------
        # Standardise to (B, H, W, C)
        # ----------------------------------------------------------------
        if x.ndim == 2:                                 # (H, W)
            x = x[np.newaxis, :, :, np.newaxis]         # (1, H, W, 1)
        elif x.ndim == 3 and x.shape[-1] not in (1, 3): # (B, H, W)
            x = x[:, :, :, np.newaxis]                  # (B, H, W, 1)
        elif x.ndim == 3:                               # (H, W, C)
            x = x[np.newaxis]                           # (1, H, W, C)
        # else x.ndim == 4: already (B, H, W, C)

        if p.ndim == 3:                                 # (H, W, 2)
            p = p[np.newaxis]                           # (1, H, W, 2)
        # else p.ndim == 4: already (B, H, W, 2)

        H, W = x.shape[1], x.shape[2]

        # ----------------------------------------------------------------
        # Branch: 128 × 128 fast path  (2-D evaluation pipeline)
        # ----------------------------------------------------------------
        if H == 128 and W == 128:
            preds = self._predict_128(x, p, batch_size, threshold)
            # Squeeze back to match the original input shape
            return self._restore_shape(preds, original_ndim_x, original_shape_x)

        # ----------------------------------------------------------------
        # Branch: tiling path  (native arbitrary resolution)
        # ----------------------------------------------------------------
        return self._predict_tiled(x, p, threshold, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Internal: 128 × 128 fast path (unchanged from original)
    # ------------------------------------------------------------------

    def _predict_128(
        self,
        x: np.ndarray,       # (B, 128, 128, 1)
        p: np.ndarray,       # (B, 128, 128, 2)
        batch_size: int,
        threshold: float,
    ) -> np.ndarray:         # (B, 128, 128, 1)
        """
        Single batched forward pass for 128 × 128 inputs.

        Always uses the JIT-compiled ``_fast_predict_fn`` — never
        ``model.predict()``.
        """
        num_samples = x.shape[0]
        chunks = []
        for start in range(0, num_samples, batch_size):
            x_chunk = tf.convert_to_tensor(x[start:start + batch_size])
            p_chunk = tf.convert_to_tensor(p[start:start + batch_size])
            chunk_logits = self._fast_predict_fn(x_chunk, p_chunk).numpy()
            chunks.append(chunk_logits)
        logits = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        return (logits >= threshold).astype(np.float32)  # (B, 128, 128, 1)

    # ------------------------------------------------------------------
    # Internal: tiling path (arbitrary resolution)
    # ------------------------------------------------------------------

    def _predict_tiled(
        self,
        x: np.ndarray,   # (B, H, W, 1)
        p: np.ndarray,   # (B, H, W, 2)
        threshold: float,
        batch_size: int = 32,
    ) -> np.ndarray:     # (B, H, W)
        """
        Tiled inference for non-128 × 128 inputs.

        Samples are processed sequentially to ensure each receives a
        unique tile grid based on its specific prompt bounding box.

        Tiles within each sample are batched (and chunked by ``batch_size``)
        to maintain high GPU utilisation while staying within memory limits.

        Returns a ``(B, H, W)`` float32 binary array at native resolution.
        """
        B, H, W, _ = x.shape
        T   = self._tiler.tile_size
        w2d = self._tiler._weight2d
        eps = 1e-8

        results = []
        for i in range(B):
            img_plane        = x[i, :, :, 0]
            prompt_img_plane = p[i, :, :, 0]
            prompt_msk_plane = p[i, :, :, 1]

            # 1. Sample-specific tile planning
            tile_starts = self._tiler._plan_tiles(prompt_msk_plane, H, W)
            n_tiles = len(tile_starts)

            # 2. Extract and chunk-batch tiles
            accum_prob   = np.zeros((H, W), dtype=np.float32)
            accum_weight = np.zeros((H, W), dtype=np.float32)

            for start_t in range(0, n_tiles, batch_size):
                end_t = min(start_t + batch_size, n_tiles)
                chunk_starts = tile_starts[start_t:end_t]
                
                img_patches    = []
                prompt_patches = []
                for y0, x0 in chunk_starts:
                    img_p  = self._tiler._extract_patch(img_plane,        y0, x0, H, W)
                    pimg_p = self._tiler._extract_patch(prompt_img_plane, y0, x0, H, W)
                    pmsk_p = self._tiler._extract_patch(prompt_msk_plane, y0, x0, H, W)
                    img_patches.append(img_p)
                    prompt_patches.append(np.stack([pimg_p, pmsk_p], axis=-1))

                # Batch forward pass
                img_batch    = np.stack(img_patches,    axis=0)[:, :, :, np.newaxis]
                prompt_batch = np.stack(prompt_patches, axis=0)
                prob_batch   = self._fast_predict_fn(
                    tf.constant(img_batch,    dtype=tf.float32),
                    tf.constant(prompt_batch, dtype=tf.float32),
                ).numpy()  # (batch_size, 128, 128, 1)

                # 3. Accumulate results for the sample
                for j, (y0, x0) in enumerate(chunk_starts):
                    prob_patch = prob_batch[j, :, :, 0]
                    y_end = min(y0 + T, H)
                    x_end = min(x0 + T, W)
                    oy, ox = y_end - y0, x_end - x0
                    accum_prob  [y0:y_end, x0:x_end] += prob_patch[:oy, :ox] * w2d[:oy, :ox]
                    accum_weight[y0:y_end, x0:x_end] += w2d[:oy, :ox]

            # 4. Final binarization for the sample
            prob = np.where(accum_weight > eps, accum_prob / (accum_weight + eps), 0.0)
            results.append((prob >= threshold).astype(np.float32))

        return np.stack(results, axis=0)  # (B, H, W)


    # ------------------------------------------------------------------
    # Internal: restore original shape for the 128×128 fast path
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_shape(
        preds: np.ndarray,      # (B, 128, 128, 1)
        original_ndim: int,
        original_shape: tuple,
    ) -> np.ndarray:
        """Squeeze ``preds`` to mirror the caller's input shape."""
        if original_ndim == 2:
            return preds[0, ..., 0]          # (128, 128)
        elif original_ndim == 3:
            if original_shape[-1] in (1, 3):
                return preds[0]              # (128, 128, 1)
            else:
                return preds[..., 0]         # (B, 128, 128)
        else:
            return preds                     # (B, 128, 128, 1)
