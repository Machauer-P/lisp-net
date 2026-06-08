"""
eval_pipeline_2d.py
===================
Evaluation pipeline comparing Prompt-UNet and UniverSeg on 2D test datasets.

Normalization convention
------------------------
All datasets are stored with TWO image representations per NPZ bundle:

  • x / sx   : z-score [-5, 5]   → used by Prompt-UNet (trained on this range)
  • x_u / sx_u : min-max [0, 1] → used by UniverSeg   (matching its training pipeline:
                 CT clip [-500,1000] → min-max; MRI 0.5–99.5 percentile → min-max)

Both normalizations are baked into the bundle at generation time from the
raw 3-D volume, so cross-slice relative brightness is always preserved.
No renormalization is needed at inference.

Dataset format
--------------
NPZ bundles (see data/test_data/ds_handler_2d.py).
Each bundle contains both the query set and a fixed 16-sample support set.
"""

import os
import sys
import time
import pickle
import psutil
import gc
import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.test_data.ds_handler_2d import load_2d_npz_bundle
from utils.metrics import dice_numpy
from evaluation.benchmark_models.UniverSeg.universeg import universeg as load_universeg_model
from inference.predictor import PromptUNetPredictor


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class EvalPipeline2D:
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Allow TF to grow GPU memory rather than pre-allocating all of it
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

    def _sync_and_clean_memory(self):
        """Force Python Garbage Collection and clear device caches between dataset runs to avoid RAM explosions."""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        mem = psutil.virtual_memory()
        if mem.percent > 90.0:
            print(f"  [ATTENTION] System RAM is critically high: {mem.percent}%.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_universeg(self, pretrained=True):
        """Load and prepare the UniverSeg model."""
        model = load_universeg_model(pretrained=pretrained)
        model.to(self.device)
        model.eval()

        # torch.compile fuses the dense CrossConv2d kernels into an optimised
        # graph (~20-50 % faster on CPU for conv-heavy models like UniverSeg).
        # Requires PyTorch >= 2.0 AND a C compiler.
        # On native Windows, Inductor requires MSVC (cl.exe). Compilation is
        # lazy (triggered on first forward pass, not here), so a try/except
        # around torch.compile() would not catch the InductorError.
        # Skip on Windows to avoid a crash mid-evaluation.
        if int(torch.__version__.split(".")[0]) >= 2 and sys.platform != "win32":
            model = torch.compile(model)
            print("[INFO] UniverSeg compiled with torch.compile.")
        else:
            print("[INFO] torch.compile skipped "
                  f"({'Windows — MSVC required' if sys.platform == 'win32' else 'PyTorch < 2.0'}). "
                  "Running UniverSeg in eager mode.")

        return model

    def load_prompt_unet(self, model_path):
        """Load a Prompt-UNet model from a local path or Hugging Face repo ID."""
        return PromptUNetPredictor(model_path)

    # ------------------------------------------------------------------
    # Dataset discovery
    # ------------------------------------------------------------------

    def discover_test_sets(self, data_path):
        """Scan data_path for NPZ bundles and return a sorted list of dicts.

        Each dict has keys:
            'index'       : str  — numeric prefix from the filename
            'name'        : str  — full stem (e.g. '3_hanseg')
            'bundle_path' : str  — absolute path to the .npz file
        """
        data_path = Path(data_path)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path

        npz_files = sorted(data_path.glob("*.npz"))
        if not npz_files:
            print(f"WARNING: No NPZ bundles found in {data_path}")
            return []

        pairs = []
        for f in npz_files:
            index = f.stem.split('_')[0]
            pairs.append({
                'index':       index,
                'name':        f.stem,
                'bundle_path': str(f),
            })

        try:
            pairs.sort(key=lambda x: int(x['index']))
        except ValueError:
            pairs.sort(key=lambda x: x['index'])

        return pairs

    # ------------------------------------------------------------------
    # Support set loading
    # ------------------------------------------------------------------

    def _load_universeg_support(self, bundle_path):
        """Load UniverSeg support images/labels from an NPZ bundle.

        Images are read from 'sx_u' which is already in [0, 1]
        (UniverSeg's training normalisation, baked in at generation time).

        Returns
        -------
        support_images : torch.Tensor  (S, 1, 128, 128)  on self.device
        support_labels : torch.Tensor  (S, 1, 128, 128)  on self.device
        """
        _, support = load_2d_npz_bundle(
            filename=Path(bundle_path).name,
            path=str(Path(bundle_path).parent),
        )

        sx_u = support['sx_u']  # (S, 128, 128, 1)  in [0, 1]
        sy   = support['sy']    # (S, 128, 128, 1)  binary {0, 1}

        if sx_u is None:
            raise ValueError(
                f"Bundle '{bundle_path}' is missing 'sx_u' key. "
                "Re-generate it with generate_2d_test_data.py."
            )

        if len(sx_u) < 16:
            print(f"  WARNING: support set has only {len(sx_u)} samples (expected 16).")

        support_images_list = []
        support_labels_list = []
        
        for i in range(len(sx_u)):
            # The input might be natively differently shaped across support pieces
            img = torch.from_numpy(sx_u[i].squeeze(-1)).float().unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, H, W)
            lbl = torch.from_numpy(sy[i].squeeze(-1)).float().unsqueeze(0).unsqueeze(0).to(self.device)   # (1, 1, H, W)
            
            # For UniverSeg, unconditionally resize everything to 128x128
            img_128 = torch.nn.functional.interpolate(img, size=(128, 128), mode='bilinear', align_corners=False)
            lbl_128 = torch.nn.functional.interpolate(lbl, size=(128, 128), mode='nearest')
            
            support_images_list.append(img_128.squeeze(0)) # (1, 128, 128)
            support_labels_list.append(lbl_128.squeeze(0)) # (1, 128, 128)

        # UniverSeg expects (S, 1, 128, 128)
        support_images = torch.stack(support_images_list, dim=0)
        support_labels = torch.stack(support_labels_list, dim=0)
        
        return support_images, support_labels

    # ------------------------------------------------------------------
    # Pair evaluation
    # ------------------------------------------------------------------

    def evaluate_pair(self, pair, model, model_name, batch_size=32, threshold=0.5):
        """Evaluate one NPZ bundle against the chosen model.

        Returns
        -------
        (mean_dice, time_taken) : (float, float)
        """
        bundle_path = pair['bundle_path']
        query, _ = load_2d_npz_bundle(
            filename=Path(bundle_path).name,
            path=str(Path(bundle_path).parent),
        )

        task_arr = query.get('task')
        task_id = int(task_arr) if task_arr is not None else None

        y_arr = query['y']   # (N, 128, 128, 1)  binary labels
        N = len(y_arr)

        dices = []
        t0 = time.time()

        if model_name == 'prompt_unet':
            x_arr = query['x']   # object array (N, arrays...)
            p_arr = query['p']   # object array (N, arrays...)
            
            for i in range(N):
                # Pass each sample individually because varying native spatial shapes 
                # prevent monolithic batching (N, H, W, ...). Prompt-UNet processes them  
                # gracefully with its native sliding-window inference layout (Adaptive Tiling).
                x_i = np.expand_dims(x_arr[i], axis=0) # (1, H, W, 1)
                p_i = np.expand_dims(p_arr[i], axis=0) # (1, H, W, 2)
                
                preds_pn = model.predict(x_i, p_i, batch_size=batch_size, threshold=threshold)
                
                dice_p = dice_numpy(y_arr[i].squeeze(), preds_pn[0].squeeze())
                dices.append(dice_p)

        elif model_name == 'universeg':
            # Prepare UniverSeg support once per bundle
            support_images, support_labels = self._load_universeg_support(bundle_path)
            
            x_u_np = query['x_u']   # object array of arrays
            
            if x_u_np is None:
                raise ValueError(
                    f"Bundle '{bundle_path}' is missing 'x_u' key. "
                    "Re-generate it with generate_2d_test_data.py."
                )
            
            # Batch size is effectively 1 since images have native independent sizes
            for i in range(N):
                x_i = torch.from_numpy(x_u_np[i].squeeze(-1)).float().unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, H, W)
                
                # UniverSeg relies heavily on whole-volume context and global relative scaling.
                # It does not perform well with isolated local patches, preventing the use 
                # of Prompt-UNet's sliding-window adaptive tiling.
                # Therefore, we utilize UniverSeg's native intended strategy for arbitrary test sizes:
                # a naive interpolation of the full cross-section into exactly 128x128.
                x_i_128 = torch.nn.functional.interpolate(x_i, size=(128, 128), mode='bilinear', align_corners=False)
                
                # Expand standard support for batch compatibility
                sup_imgs_batch = support_images.unsqueeze(0) # (1, S, 1, 128, 128)
                sup_lbls_batch = support_labels.unsqueeze(0) # (1, S, 1, 128, 128)

                with torch.no_grad():
                    logits = model(x_i_128, sup_imgs_batch, sup_lbls_batch) # (1, 1, 128, 128)
                    
                    # To compare FAIRLY with the native-resolution ground truth, we resize the 
                    # 128x128 logit prediction back up to the native shape (H, W).
                    native_shape = (x_u_np[i].shape[0], x_u_np[i].shape[1])
                    logits_native = torch.nn.functional.interpolate(logits, size=native_shape, mode='bilinear', align_corners=False)
                    
                    u_preds = torch.sigmoid(logits_native.squeeze()) # (H, W)
                    u_preds = (u_preds > threshold).float().cpu().numpy()

                dice_u = dice_numpy(y_arr[i].squeeze(), u_preds)
                dices.append(dice_u)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        t1 = time.time()
        time_taken = t1 - t0

        return np.mean(dices), time_taken, task_id

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_evaluation(self, data_path, model_name, p_unet_model=None, batch_size=32, output_file=None):
        """Discover all NPZ bundles, run inference, and print results.

        Parameters
        ----------
        data_path      : str or Path  — directory containing NPZ bundles
        model_name     : str          — 'prompt_unet' or 'universeg'
        p_unet_model   : str or None  — local .keras path or Hugging Face repo ID
        output_file    : str or None  — if given, pickle results here

        Returns
        -------
        list of dicts with keys 'index', 'name', 'model', 'dice', 'time'
        """
        print(f"Starting evaluation on: {data_path}")
        pairs = self.discover_test_sets(data_path)
        print(f"Discovered {len(pairs)} dataset(s).")

        # On CPU, float16 has no hardware acceleration. Force float32 so mixed-precision
        # models (e.g. v292) run at native CPU speed instead of emulated float16.
        if len(tf.config.list_physical_devices('GPU')) == 0:
            tf.keras.mixed_precision.set_global_policy('float32')
            print("[INFO] CPU detected — forcing float32 compute policy.")

        print(f"Loading '{model_name}' model...")
        if model_name == 'prompt_unet':
            model = self.load_prompt_unet(p_unet_model)
        elif model_name == 'universeg':
            model = self.load_universeg()
        else:
            raise ValueError("model_name must be 'prompt_unet' or 'universeg'")

        results = []
        for pair in tqdm(pairs, desc=f"Evaluating {model_name}"):
            dice_mean, time_taken, task_id = self.evaluate_pair(pair, model, model_name, batch_size=batch_size)
            results.append({
                'index': pair['index'],
                'name':  pair['name'],
                'model': model_name,
                'dice':  dice_mean,
                'time':  time_taken,
                'task':  task_id,
            })

            # Ensure memory traces from last pair are purged
            self._sync_and_clean_memory()

        if results:
            avg_dice = np.mean([r['dice'] for r in results])
            avg_time = np.mean([r['time'] for r in results])

            print(f"\nFinal Results for {model_name}:")
            if model_name == 'prompt_unet':
                print(f"  Prompt U-Net ({p_unet_model or 'HF default'}) Mean Dice : {avg_dice:.4f}  (Avg Time: {avg_time:.2f}s/dataset)")
            else:
                print(f"  UniverSeg               Mean Dice : {avg_dice:.4f}  (Avg Time: {avg_time:.2f}s/dataset)")

        if output_file:
            if not results:
                print(f"WARNING: No results to save (no NPZ bundles found). "
                      f"Output file '{output_file}' was NOT overwritten.")
            else:
                with open(output_file, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Results saved to {output_file}")

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run evaluation on 2D test datasets for either Prompt-UNet or UniverSeg."
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["prompt_unet", "universeg"],
        help="Which model to evaluate: 'prompt_unet' or 'universeg'.",
    )
    parser.add_argument(
        "--data_path", type=str,
        default="data/test_data/2d/offset_5",
        help="Directory containing NPZ bundles.",
    )
    parser.add_argument(
        "--p_unet_model", type=str, default=None,
        help="Prompt-UNet model: local .keras path or Hugging Face repo ID (default: HF).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for Prompt-UNet tiled inference path.",
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Path to save pickled results. Defaults to evaluation/benchmark_universeg/eval_results_<model>.pkl",
    )
    args = parser.parse_args()

    if not args.output:
        args.output = f"evaluation/benchmark_universeg/eval_results_{args.model}.pkl"

    pipeline = EvalPipeline2D()
    pipeline.run_full_evaluation(args.data_path, args.model, args.p_unet_model, args.batch_size, args.output)
