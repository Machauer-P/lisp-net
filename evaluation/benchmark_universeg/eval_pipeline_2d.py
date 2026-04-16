"""
eval_pipeline_2d.py
===================
Evaluation pipeline comparing Prompt-UNet and UniverSeg on 2D test datasets.

Normalization convention
------------------------
All datasets are stored with images in the z-score range [-5, 5]
(p_unet_292 training convention).  At inference:

  • Prompt-UNet  → receives images in [-5, 5] (no renorm needed)
  • UniverSeg    → receives images renorm'd to [0, 1] via (x + 5) / 10
                   (required by the official UniverSeg repo)

Dataset format
--------------
NPZ bundles (see data/test_data/ds_handler_2d.py).
Each bundle contains both the query set and a fixed 16-sample support set.
"""

import os
import sys
import pickle
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


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _renorm_for_universeg(x):
    """Map z-score range [-5, 5] → UniverSeg-required [0, 1]."""
    return np.clip((np.asarray(x, dtype=np.float32) + 5.0) / 10.0, 0.0, 1.0)


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

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_universeg(self, pretrained=True):
        """Load and prepare the UniverSeg model."""
        model = load_universeg_model(pretrained=pretrained)
        model.to(self.device)
        model.eval()
        return model

    def load_prompt_unet(self, model_version='292'):
        """Load a saved Prompt-UNet .keras model."""
        model_path = PROJECT_ROOT / f'training/p_unet_{model_version}.keras'
        if not model_path.exists():
            raise FileNotFoundError(
                f"Prompt U-Net model not found at {model_path}"
            )
        return tf.keras.models.load_model(str(model_path))

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
        """Load support images/labels from an NPZ bundle and prepare them
        for UniverSeg inference.

        Images are renormalised from [-5, 5] → [0, 1].

        Returns
        -------
        support_images : torch.Tensor  (S, 1, 128, 128)  on self.device
        support_labels : torch.Tensor  (S, 1, 128, 128)  on self.device
        """
        _, support = load_2d_npz_bundle(
            filename=Path(bundle_path).name,
            path=str(Path(bundle_path).parent),
        )

        sx = support['sx']  # (S, 128, 128, 1)  in [-5, 5]
        sy = support['sy']  # (S, 128, 128, 1)  binary {0, 1}

        if len(sx) < 16:
            print(f"  WARNING: support set has only {len(sx)} samples (expected 16).")

        # Renorm images to [0, 1] for UniverSeg; labels are already binary
        sx_u = _renorm_for_universeg(sx)          # (S, 128, 128, 1)

        # UniverSeg expects (S, 1, H, W)
        support_images = (
            torch.from_numpy(sx_u.squeeze(-1))    # (S, 128, 128)
            .float().unsqueeze(1)                  # (S, 1, 128, 128)
            .to(self.device)
        )
        support_labels = (
            torch.from_numpy(sy.squeeze(-1))
            .float().unsqueeze(1)
            .to(self.device)
        )
        return support_images, support_labels

    # ------------------------------------------------------------------
    # Pair evaluation
    # ------------------------------------------------------------------

    def evaluate_pair(self, pair, u_seg_model, p_unet_model, threshold=0.45):
        """Evaluate one NPZ bundle against both models.

        Returns
        -------
        (mean_dice_prompt_unet, mean_dice_universeg) : (float, float)
        """
        bundle_path = pair['bundle_path']
        query, _ = load_2d_npz_bundle(
            filename=Path(bundle_path).name,
            path=str(Path(bundle_path).parent),
        )

        x_arr = query['x']   # (N, 128, 128, 1)  images in [-5, 5]
        y_arr = query['y']   # (N, 128, 128, 1)  binary labels
        p_arr = query['p']   # (N, 128, 128, 2)  prompts

        # Prepare UniverSeg support once per bundle
        support_images, support_labels = self._load_universeg_support(bundle_path)

        prompt_unet_dices = []
        universeg_dices   = []

        for i in range(len(x_arr)):
            x = x_arr[i]  # (128, 128, 1)  in [-5, 5]
            y = y_arr[i]  # (128, 128, 1)  binary
            p = p_arr[i]  # (128, 128, 2)

            # ---- Prompt-UNet inference ----
            # Input in [-5, 5] — exactly the training range, no renorm needed
            x_in = x[np.newaxis]   # (1, 128, 128, 1)
            p_in = p[np.newaxis]   # (1, 128, 128, 2)

            pred_pn = p_unet_model.predict([x_in, p_in], verbose=0)  # (1, 128, 128, 1)
            pred_pn = (pred_pn >= threshold).astype(np.float32)

            # Both squeezed to (128, 128) before dice to avoid any axis confusion
            dice_p = dice_numpy(y.squeeze(), pred_pn.squeeze())
            prompt_unet_dices.append(dice_p)

            # ---- UniverSeg inference ----
            # Renorm to [0, 1] as required by the official model
            x_u_np = _renorm_for_universeg(x)                      # (128, 128, 1)
            x_u = (
                torch.from_numpy(x_u_np.squeeze(-1))               # (128, 128)
                .float()
                .unsqueeze(0).unsqueeze(0)                         # (1, 1, 128, 128)
                .to(self.device)
            )

            with torch.no_grad():
                # Model signature:
                #   model(target_image, support_images, support_labels)
                #   target_image  : (B, 1, H, W)
                #   support_images: (B, S, 1, H, W)  ← add batch dim with [None]
                #   support_labels: (B, S, 1, H, W)
                #   → returns     : (B, 1, H, W)
                logits = u_seg_model(
                    x_u,
                    support_images[None],   # (1, S, 1, 128, 128)
                    support_labels[None],   # (1, S, 1, 128, 128)
                )                           # → (1, 1, 128, 128)

                # Sigmoid + threshold; collapse to (128, 128)
                u_pred = torch.sigmoid(logits[0, 0])           # (128, 128)
                u_pred = (u_pred > threshold).float().cpu().numpy()

            dice_u = dice_numpy(y.squeeze(), u_pred)           # both (128, 128)
            universeg_dices.append(dice_u)

        return np.mean(prompt_unet_dices), np.mean(universeg_dices)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_evaluation(self, data_path, p_unet_version='292', output_file=None):
        """Discover all NPZ bundles, run inference, and print results.

        Parameters
        ----------
        data_path      : str or Path  — directory containing NPZ bundles
        p_unet_version : str          — model version suffix (e.g. '292')
        output_file    : str or None  — if given, pickle results here

        Returns
        -------
        list of dicts with keys 'index', 'name', 'prompt_unet_dice', 'universeg_dice'
        """
        print(f"Starting evaluation on: {data_path}")
        pairs = self.discover_test_sets(data_path)
        print(f"Discovered {len(pairs)} dataset(s).")

        print("Loading models...")
        u_seg  = self.load_universeg()
        p_unet = self.load_prompt_unet(p_unet_version)

        results = []
        for pair in tqdm(pairs, desc="Evaluating"):
            dice_p, dice_u = self.evaluate_pair(pair, u_seg, p_unet)
            results.append({
                'index':            pair['index'],
                'name':             pair['name'],
                'prompt_unet_dice': dice_p,
                'universeg_dice':   dice_u,
            })

        avg_p = np.mean([r['prompt_unet_dice'] for r in results])
        avg_u = np.mean([r['universeg_dice']   for r in results])

        print(f"\nFinal Results:")
        print(f"  Prompt U-Net (V{p_unet_version}) Mean Dice : {avg_p:.4f}")
        print(f"  UniverSeg               Mean Dice : {avg_u:.4f}")

        if output_file:
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
        description="Compare Prompt-UNet and UniverSeg on 2D test datasets."
    )
    parser.add_argument(
        "--data_path", type=str,
        default="data/test_data/2d/offset_5",
        help="Directory containing NPZ bundles.",
    )
    parser.add_argument(
        "--p_unet_version", type=str, default="292",
        help="Prompt-UNet model version suffix.",
    )
    parser.add_argument(
        "--output", type=str,
        default="evaluation/benchmark_universeg/eval_results.pkl",
        help="Path to save pickled results.",
    )
    args = parser.parse_args()

    pipeline = EvalPipeline2D()
    pipeline.run_full_evaluation(args.data_path, args.p_unet_version, args.output)
