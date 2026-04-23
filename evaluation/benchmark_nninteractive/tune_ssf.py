"""
evaluation/benchmark_nninteractive/tune_ssf.py
==============================================
This script isolates and evaluates the Self-Supervised Feedback (SSF) propagation 
mechanism of Prompt-UNet across a battery of threshold configurations.
It bypasses the Interactive Feedback Loop (IFL) and nnInteractive completely for 
vastly increased execution speed.

USE THIS ON YOUR TRAINING DATA: 
Run this script to empirically determine the optimal `sim_diff_threshold` parameter 
based on random samplings of your training datasets. (Do NOT use this on test data! 
Unfair, because in real world scenario you do not have access to the test data labels.)
0.0 represents the baseline (SSF completely disabled). Once you identify the threshold
yielding the highest volumetric Dice scores, lock that value into your main 
`benchmark_3d.py` evaluations.

Usage (CLI):
    python evaluation/benchmark_nninteractive/tune_ssf.py
"""

import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.test_data.ds_handler import load_dataset
from utils.metrics import volumetric_dice
from inference.inference_volume import VolumeInference, generate_initial_prompt


def tune_ssf():
    # Configuration
    npz_paths = [
        "data/test_data/han_seg_ct.npz",
        "data/test_data/han_seg_mri.npz",
        "data/test_data/SegRap2023.npz",
        "data/test_data/HCCTase_ceCT.npz",
    ]
    model_path = "training/p_unet_313.keras"
    thresholds = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    runs_per_vol = 10
    subset_ratio = 0.30

    print(f"\n========================================================")
    print(f" SSF TUNING SCRIPT: {model_path} ")
    print(f"========================================================")

    model_full_path = _PROJECT_ROOT / model_path
    if not model_full_path.exists():
        print(f"Model not found at {model_full_path}. Please check path.")
        return

    # Load Model (JIT compilation triggers once)
    print("Loading core architecture in background...")
    p_unet = VolumeInference(
        model_path=str(model_full_path),
        modality="MRI",  # Replaced dynamically per volume
        output_threshold=0.45,
    )

    all_records = []

    for path_str in npz_paths:
        path = _PROJECT_ROOT / path_str
        if not path.exists():
            print(f"\nWarning: Dataset '{path.name}' not found. Skipping.")
            continue
        
        print(f"\n[{path.name}] Extracting data...")
        dataset = load_dataset(str(path))
        dataset_name = path.stem
        
        pids = list(dataset.keys())
        pids.sort()
        # Ensure deterministic shuffling 
        random.seed(42)
        random.shuffle(pids)
        
        num_vols = max(1, int(len(pids) * subset_ratio))
        selected_pids = pids[:num_vols]
        
        print(f"[{path.name}] Subsampling 30%: Using {num_vols} out of {len(pids)} volumes.")
        
        # Evaluate each selected volume
        for pid in selected_pids:
            item = dataset[pid]
            img_3d   = np.asarray(item["image"]).astype(np.float32)
            segs     = item["segmentations"]
            vol_mod  = item.get("modality", "MRI") 

            # Combine multi-class to unified integer foreground labels
            if isinstance(segs, list):
                if len(segs) == 0:
                    continue
                seg_3d_labels = np.zeros_like(img_3d, dtype=np.int32)
                for label_idx, seg_arr in enumerate(segs, start=1):
                    seg_3d_labels[np.asarray(seg_arr) != 0] = label_idx
            else:
                seg_3d_labels = np.asarray(segs).astype(np.int32)

            if np.all(seg_3d_labels == 0):
                continue
                
            print(f"  → Vol: {pid} | Modality: {vol_mod} | Shape: {img_3d.shape}")
            
            # Execute repetitive runs per volume
            for run_idx in range(runs_per_vol):
                # 1) Generate the shared geometric origin prompt.
                # Must catch edge case where volume has insufficient min_pixels 
                try:
                    initial_prompt_3d, initial_prompt_2d_seg, (prompt_axis, prompt_idx), selected_roi = \
                        generate_initial_prompt(seg_3d_labels, min_pixels=50)
                except Exception as e:
                    print(f"    [Run {run_idx+1}] Skipping: {e}")
                    continue
                    
                seg_3d_binary = (seg_3d_labels == selected_roi).astype(np.float32)
                print(f"    * Run {run_idx+1}/{runs_per_vol} [Axis={prompt_axis}, Slice={prompt_idx}] ", end="")
                
                # 2) Sweep threshold parameters across the SAME prompt constraint
                run_metrics = []
                for thresh in thresholds:
                    # Dynamically alter SSF parameter on live instance to circumvent recompilation time 
                    p_unet.sim_diff_threshold = thresh
                    
                    t0 = time.perf_counter()
                    result = p_unet.run(
                        img_3d=img_3d,
                        seg_3d_binary=seg_3d_binary,
                        initial_prompt_2d_seg=initial_prompt_2d_seg,
                        prompt_axis=prompt_axis,
                        prompt_idx=prompt_idx,
                        modality=vol_mod,
                    )
                    t_eval = time.perf_counter() - t0
                    
                    # Manual volumetric reconstruction matching benchmark behaviour 
                    pred_vol = np.zeros_like(img_3d, dtype=np.float32)
                    ordered_indices = list(reversed(result.backward_indices)) + [result.prompt_idx] + result.forward_indices
                    slices_2d = result.results_3d
                    if slices_2d.ndim == 2:
                        slices_2d = slices_2d[np.newaxis]
                        
                    for local_i, v_idx in enumerate(ordered_indices):
                        if local_i >= len(slices_2d): 
                            break
                        s = slices_2d[local_i]
                        
                        # Prevent sizing collisions manually 
                        if prompt_axis == 0:
                            pred_vol[v_idx] = s
                        elif prompt_axis == 1:
                            pred_vol[:, v_idx, :] = s
                        else:
                            pred_vol[:, :, v_idx] = s
                            
                    vol_dice = float(volumetric_dice(seg_3d_binary, pred_vol))
                    
                    record = {
                        "dataset": dataset_name,
                        "pid": pid,
                        "run_idx": run_idx,
                        "threshold": thresh,
                        "vol_dice": vol_dice,
                        "eval_time_s": t_eval
                    }
                    all_records.append(record)
                    run_metrics.append(f"{thresh}={vol_dice:.3f}")
                
                print(" | ".join(run_metrics))

    # Compile Final Tabular Output representation
    if not all_records:
        print("\nNo statistics could be collected. Aborting result generation.")
        return 

    df = pd.DataFrame(all_records)
    
    print("\n\n" + "="*70)
    print(" SSF HYPERPARAMETER TUNING SUMMARY")
    print("="*70)

    print("\n1. Overall Performance per Threshold Sequence")
    summary_overall = df.groupby("threshold").agg(
        mean_dice=("vol_dice", "mean"),
        std_dice=("vol_dice", "std"),
        mean_time_s=("eval_time_s", "mean"),
        num_runs=("vol_dice", "count")
    ).round(3)
    print(summary_overall)
    
    print("\n" + "-"*70)
    print("2. Dice Score by Dataset Matrix (Thresholds applied globally)")
    summary_ds = df.groupby(["dataset", "threshold"])["vol_dice"].mean().unstack().round(3)
    print(summary_ds)
    print("="*70 + "\n")
    
    # Save standard output csv to execution dictory 
    out_csv = _HERE / "ssf_tuning_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"[*] Deep dive statistical traces written permanently to:\n    {out_csv}\n")

if __name__ == "__main__":
    tune_ssf()
