"""
inference/tune_ssf.py
======================
Unified SSF hyperparameter tuning script for Prompt-UNet.

This script compares three Self-Supervised Feedback (SSF) strategies across
a 30% random subset of your TRAINING datasets, running 10 inference passes
per volume with the same geometric prompt.

USE THIS ON YOUR TRAINING DATA:
Run this script to empirically determine the best SSF strategy and threshold
for your model. DO NOT use test data — in a real deployment scenario you have
no access to test-set labels, so tuning on them would be unfair.

Once you identify the best strategy + threshold, lock it into your main
benchmark_3d.py evaluations:

    from inference.ssf import RelativeSSIMStrategy
    run_benchmark(..., ssf_strategy=RelativeSSIMStrategy(0.25))

Strategies tested
-----------------
  none             — SSF disabled; serves as the baseline.
  RelativeSSIM     — fires when SSIM drops by X% from start (dataset-invariant).
  MaskDice         — fires when consecutive mask Dice < threshold.
  ConfidenceDrop   — fires when mean foreground sigmoid drops by X% from start.

Usage (CLI):
    python inference/tune_ssf.py
"""

import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root injection
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.test_data.ds_handler import load_dataset
from utils.metrics import volumetric_dice
from inference.inference_volume import VolumeInference, generate_initial_prompt
from inference.ssf import (
    RelativeSSIMStrategy,
    MaskDiceStrategy,
    ConfidenceDropStrategy,
)


# ---------------------------------------------------------------------------
# Strategy sweep definition
# ---------------------------------------------------------------------------
# Each entry: (label, strategy_instance_or_None)
# None = SSF disabled → baseline
STRATEGIES_TO_TEST = [
    # --- Baseline ---
    ("none",                   None),
    # --- RelativeSSIM ---
    ("RelSSIM(0.10)",          RelativeSSIMStrategy(0.10)),
    ("RelSSIM(0.20)",          RelativeSSIMStrategy(0.20)),
    ("RelSSIM(0.30)",          RelativeSSIMStrategy(0.30)),
    ("RelSSIM(0.40)",          RelativeSSIMStrategy(0.40)),
    # --- MaskDice ---
    ("MaskDice(0.20)",         MaskDiceStrategy(0.20)),
    ("MaskDice(0.30)",         MaskDiceStrategy(0.30)),
    ("MaskDice(0.40)",         MaskDiceStrategy(0.40)),
    ("MaskDice(0.50)",         MaskDiceStrategy(0.50)),
    ("MaskDice(0.60)",         MaskDiceStrategy(0.60)),
    ("MaskDice(0.70)",         MaskDiceStrategy(0.70)),
    # --- ConfidenceDrop ---
    ("Confidence(0.05)",       ConfidenceDropStrategy(0.05)),
    ("Confidence(0.10)",       ConfidenceDropStrategy(0.10)),
    ("Confidence(0.20)",       ConfidenceDropStrategy(0.20)),
    ("Confidence(0.30)",       ConfidenceDropStrategy(0.30)),
    ("Confidence(0.40)",       ConfidenceDropStrategy(0.40)),
]


def tune_ssf():
    """Run the SSF strategy sweep and print a comparison table."""
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    npz_paths = [
        "data/train_data/nako_combined.npz",
        "data/train_data/total_seg_combined.npz",
        "data/train_data/msd_combined.npz",
        "data/train_data/brats_gli.npz",
        "data/train_data/brats_men_rt.npz",
    ]
    model_path   = "training/p_unet_313.keras"
    runs_per_vol = 10
    subset_ratio = 0.10
    batch_size   = 3     # Increase for large GPUs (e.g. 8 or 12)
    buffer_size  = 6     # Number of recent predictions to aggregate for SSF refresh

    print(f"\n{'='*60}")
    print(f" SSF STRATEGY COMPARISON: {model_path}")
    print(f" Strategies: {len(STRATEGIES_TO_TEST)}  |  Runs/vol: {runs_per_vol}")
    print(f"{'='*60}\n")

    model_full_path = _PROJECT_ROOT / model_path
    if not model_full_path.exists():
        print(f"[ERROR] Model not found: {model_full_path}")
        return

    # Load model once; swap strategy via set_ssf_strategy() between runs
    p_unet = VolumeInference(
        model_path       = str(model_full_path),
        modality         = None,    # we pass it per-volume in run()
        output_threshold = 0.5,
        ssf_strategy     = None,    # start disabled; overridden in loop
        buffer_size      = buffer_size,
        batch_size       = batch_size,
    )

    all_records = []

    for path_str in npz_paths:
        path = _PROJECT_ROOT / path_str
        if not path.exists():
            print(f"[WARN] Dataset not found: {path.name}  →  skipping.")
            continue

        print(f"\n── {path.name} ──────────────────────────────────────")
        dataset      = load_dataset(str(path))
        dataset_name = path.stem

        pids = list(dataset.keys())
        pids.sort()
        random.seed(42)
        random.shuffle(pids)

        num_vols      = max(1, int(len(pids) * subset_ratio))
        selected_pids = pids[:num_vols]
        print(f"  Subsampling {subset_ratio:.0%}: {num_vols}/{len(pids)} volumes")

        for pid in selected_pids:
            item    = dataset[pid]
            img_3d  = np.asarray(item["image"]).astype(np.float32)
            segs    = item["segmentations"]
            vol_mod = item.get("modality", "MRI")

            # Build integer label volume
            if isinstance(segs, list):
                if not segs:
                    continue
                seg_3d_labels = np.zeros_like(img_3d, dtype=np.int32)
                for lbl_idx, seg_arr in enumerate(segs, start=1):
                    seg_3d_labels[np.asarray(seg_arr) != 0] = lbl_idx
            else:
                seg_3d_labels = np.asarray(segs).astype(np.int32)

            if np.all(seg_3d_labels == 0):
                continue

            print(f"\n  Volume: {pid}  |  Modality: {vol_mod}  |  Shape: {img_3d.shape}")

            for run_idx in range(runs_per_vol):
                # Generate a random prompt (same across all strategies for fair comparison)
                try:
                    _, initial_prompt_2d_seg, (prompt_axis, prompt_idx), selected_roi = \
                        generate_initial_prompt(seg_3d_labels, min_pixels=50)
                except Exception as exc:
                    print(f"    Run {run_idx+1}: skipping — {exc}")
                    continue

                seg_3d_binary = (seg_3d_labels == selected_roi).astype(np.float32)

                print(f"    Run {run_idx+1:2d}/{runs_per_vol}"
                      f" [ax={prompt_axis} sl={prompt_idx}]  ", end="", flush=True)

                row_metrics = []

                for label, strategy in STRATEGIES_TO_TEST:
                    # Swap the strategy without reloading the model
                    p_unet.set_ssf_strategy(strategy)

                    t0     = time.perf_counter()
                    result = p_unet.run(
                        img_3d                = img_3d,
                        seg_3d_binary         = seg_3d_binary,
                        initial_prompt_2d_seg = initial_prompt_2d_seg,
                        prompt_axis           = prompt_axis,
                        prompt_idx            = prompt_idx,
                        modality              = vol_mod,
                    )
                    t_eval = time.perf_counter() - t0

                    # Reconstruct full volume prediction
                    pred_vol     = np.zeros_like(img_3d, dtype=np.float32)
                    back_ordered = list(reversed(result.backward_indices))
                    ordered_idxs = back_ordered + [result.prompt_idx] + result.forward_indices
                    slices_2d    = result.results_3d

                    for local_i, v_idx in enumerate(ordered_idxs):
                        if local_i >= len(slices_2d):
                            break
                        s = slices_2d[local_i]
                        if prompt_axis == 0:
                            pred_vol[v_idx]       = s
                        elif prompt_axis == 1:
                            pred_vol[:, v_idx, :] = s
                        else:
                            pred_vol[:, :, v_idx] = s

                    vol_dice = float(volumetric_dice(seg_3d_binary, pred_vol))

                    all_records.append({
                        "dataset"  : dataset_name,
                        "pid"      : pid,
                        "run_idx"  : run_idx,
                        "strategy" : label,
                        "vol_dice" : vol_dice,
                        "time_s"   : t_eval,
                    })
                    row_metrics.append(f"{label}={vol_dice:.3f}")

                print("  ".join(row_metrics))

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    if not all_records:
        print("\n[WARN] No records collected.")
        return

    df = pd.DataFrame(all_records)

    print(f"\n\n{'='*70}")
    print(" SSF STRATEGY TUNING SUMMARY")
    print(f"{'='*70}")

    print("\n1. Overall mean Dice per strategy (all datasets / volumes / runs)")
    overall = df.groupby("strategy").agg(
        mean_dice=("vol_dice", "mean"),
        std_dice =("vol_dice", "std"),
        mean_time=("time_s",   "mean"),
        n        =("vol_dice", "count"),
    ).round(4)
    # Sort so best Dice is at the top
    overall = overall.sort_values("mean_dice", ascending=False)
    print(overall.to_string())

    print(f"\n{'-'*70}")
    print("2. Mean Dice by dataset × strategy (pivot table)")
    pivot = (df.groupby(["strategy", "dataset"])["vol_dice"]
               .mean()
               .unstack()
               .round(3))
    # Order rows by descending overall mean
    row_order = overall.index.tolist()
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    print(pivot.to_string())

    print(f"\n{'='*70}")

    # Identify the best strategy
    best_label = overall["mean_dice"].idxmax()
    best_dice  = overall.loc[best_label, "mean_dice"]
    baseline   = overall.loc["none", "mean_dice"] if "none" in overall.index else float("nan")
    print(f"\n  Best strategy : {best_label}  (mean Dice {best_dice:.4f})")
    print(f"  Baseline none : mean Dice {baseline:.4f}")
    print(f"  SSF gain      : {best_dice - baseline:+.4f}")

    # Save detailed CSV
    out_csv = _HERE / "ssf_tuning_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[*] Full per-run trace saved to:\n    {out_csv}\n")


if __name__ == "__main__":
    tune_ssf()
