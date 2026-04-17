import numpy as np
import tensorflow as tf
from pathlib import Path

class PromptUNetPredictor:
    """
    Fast inference wrapper for Prompt-UNet.
    Handles dimensions automatically, ensuring batch and channel axes exist
    before passing to the model, and restoring the original shape upon return.
    """
    def __init__(self, model_path_or_obj):
        """
        Initialization.
        
        Args:
            model_path_or_obj: Can be a path to a keras model (.keras or directory),
                               or a pre-loaded tf.keras.Model.
        """
        if isinstance(model_path_or_obj, (str, Path)):
            self.model = tf.keras.models.load_model(str(model_path_or_obj))
        else:
            self.model = model_path_or_obj
            
        # input_signature locks tensor shapes so TF never re-traces when the
        # batch size changes between bundles.  None = dynamic batch axis.
        # H/W/C are fixed to the 128×128 grid used throughout the pipeline.
        self._fast_predict_fn = tf.function(
            self._fast_signature,
            input_signature=[
                tf.TensorSpec([None, 128, 128, 1], tf.float32),  # image
                tf.TensorSpec([None, 128, 128, 2], tf.float32),  # prompt
            ],
        )

    def _fast_signature(self, x, p):
        """
        Wrapped function for direct tensor execution. Bypasses Keras data 
        pipeline overhead for very large speedups on single items / small batches.
        
        IMPORTANT: Never falls through to model.predict() which re-creates a 
        tf.data pipeline on every call — that overhead dominates for chunk-by-
        chunk evaluation loops (once per 64-item batch = ~0.5s wasted per call).
        """
        return self.model([x, p], training=False)

    def predict(self, image, prompt, batch_size=32, threshold=0.5):
        """
        Predict segmentation given image(s) and prompt(s).
        
        Args:
            image: numpy array or tensor of shape (H, W), (H, W, C), (B, H, W), or (B, H, W, C)
            prompt: numpy array or tensor of shape matching the batch size, with 2 channels. 
                    e.g. (H, W, 2), (B, H, W, 2)
            batch_size: Int, used to chunk batches to avoid Out-Of-Memory errors.
            threshold: Float, cutoff threshold for probability masking.
            
        Returns:
            binary_mask: np.float32 array of the same base shape as `image`.
                         Channels dimension is preserved if it existed, otherwise stripped.
        """
        x = np.asarray(image, dtype=np.float32)
        p = np.asarray(prompt, dtype=np.float32)
        
        original_ndim_x = x.ndim
        original_shape_x = x.shape
        
        # 1. Standardize Dimensions to (B, H, W, C)
        # Handle Channel dimension for Image
        if x.ndim == 2:
            x = x[..., np.newaxis] # (H, W) -> (H, W, 1)
        elif x.ndim == 3 and x.shape[-1] not in (1, 3):
            # Probably (B, H, W)
            x = x[..., np.newaxis]
            
        # Handle Channel dimension for Prompt
        # Normally prompt is strictly 2 channels as the final axis.
        # Ensure it has a batch dimension if image does.

        # Ensure both arrays have batch dimensions
        if x.ndim == 3: # Either (H, W, C) after channel fix
            x = x[np.newaxis, ...]
            
        if p.ndim == 3 and p.shape[-1] == 2:
            # (H, W, 2) -> (1, H, W, 2)
            p = p[np.newaxis, ...]

        # 2. Fastest possible inference execution
        # Always use the tf.function path and manually chunk to avoid OOM.
        # model.predict() creates a new tf.data pipeline + callbacks on EVERY call,
        # which adds ~200-500ms overhead per call — fatal when called per-batch in a loop.
        num_samples = x.shape[0]
        chunks = []
        for start in range(0, num_samples, batch_size):
            x_chunk = tf.convert_to_tensor(x[start:start + batch_size])
            p_chunk = tf.convert_to_tensor(p[start:start + batch_size])
            chunk_logits = self._fast_predict_fn(x_chunk, p_chunk).numpy()
            chunks.append(chunk_logits)
        logits = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
            
        # 3. Post-process (binarize masks)
        preds = (logits >= threshold).astype(np.float32)
        
        # 4. Squeeze to functionally mirror user input tensor shape
        if original_ndim_x == 2:
            preds = preds[0, ..., 0]   # (H, W)
        elif original_ndim_x == 3:
            if original_shape_x[-1] in (1, 3):
                # Started as (H, W, C), return (H, W, 1)
                preds = preds[0]
            else:
                # Started as (B, H, W), return (B, H, W)
                preds = preds[..., 0]
        else:
            # original was (B, H, W, C). return (B, H, W, 1) without modifying original structural integrity.
            pass
            
        return preds
