"""
evaluation/benchmark_nninteractive/run_3d_benchmark.py
=======================================================
Standalone benchmark script comparing Prompt-UNet and nnInteractive on 3-D
volumes loaded from .npz files.

Inputs
------
* One or more .npz files in the standard ds_handler format
  (created by data/test_data/ds_handler.py → save_dataset()).
* A saved Prompt-UNet .keras model.
* A directory containing nnInteractive model weights.

Outputs
-------
results_<timestamp>.pkl   – Full per-run result list (pickle).
results_<timestamp>.json  – Human-readable summary statistics.

Usage (CLI)
-----------
    python evaluation/benchmark_nninteractive/run_3d_benchmark.py \\
        --npz_paths data/test_data/Mouse_no_trachea.npz \\
        --p_unet_model training/p_unet_313.keras \\
        --nn_model_dir /path/to/nnInteractive_v1.0 \\
        --runs_per_vol 5 \\
        --mode ifl            # 'ssf' or 'ifl'

Usage (from notebook)
---------------------
    from evaluation.benchmark_nninteractive.run_3d_benchmark import run_benchmark

    records = run_benchmark(
        npz_paths       = ["data/test_data/Mouse_no_trachea.npz"],
        p_unet_model    = "training/p_unet_313.keras",
        nn_model_dir    = "/path/to/nnInteractive_v1.0",
        runs_per_vol    = 5,
        mode            = "ifl",
        output_dir      = "evaluation/benchmark_nninteractive/results",
    )
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.test_data.ds_handler import load_dataset
from utils.metrics import volumetric_dice, dice_window_prompt
from inference.inference_volume import (
    VolumeInference,
    InteractiveFeedbackLoop,
    generate_initial_prompt,
    RunResult,
)
from inference.ssf import (
    BaseSSFStrategy,
    RelativeSSIMStrategy,
    MaskDiceStrategy,
    ConfidenceDropStrategy,
)
from evaluation.benchmark_nninteractive.nninteractive_inference import NNInteractiveInference


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _reconstruct_volume(
    result,
    vol_shape: tuple,
) -> np.ndarray:
    """
    Rebuild a full 3-D binary prediction volume from a RunResult.

    Slices that were not evaluated (empty in the ground-truth) remain zero.
    This is needed for a fair volumetric Dice comparison against nnInteractive,
    which produces a full-volume prediction.

    Parameters
    ----------
    result     : RunResult from VolumeInference or InteractiveFeedbackLoop.
                 ``result.results_3d`` has shape ``(S, H_slice, W_slice)`` at
                 native volume resolution — no resize is required.
    vol_shape  : (X, Y, Z) — original volume shape.

    Returns
    -------
    np.ndarray (X, Y, Z), float32, binary {0, 1}.
    """
    pred_vol = np.zeros(vol_shape, dtype=np.float32)

    # Evaluation order: [backward_reversed..., prompt_slice, forward...]
    ordered_indices = (
        list(reversed(result.backward_indices))
        + [result.prompt_idx]
        + result.forward_indices
    )

    slices_2d = result.results_3d  # shape (S, H, W) — native resolution
    if slices_2d.ndim == 2:
        slices_2d = slices_2d[np.newaxis]  # ensure (S, H, W)

    axis = result.prompt_axis

    for local_i, vol_idx in enumerate(ordered_indices):
        if local_i >= len(slices_2d):
            break
        s = slices_2d[local_i]  # (H_slice, W_slice) — already native resolution

        # Direct assignment — no resize needed
        if axis == 0:
            pred_vol[vol_idx] = s
        elif axis == 1:
            pred_vol[:, vol_idx, :] = s
        else:
            pred_vol[:, :, vol_idx] = s

    return pred_vol


def _make_run_record(
    volume_id: str,
    pid: str,
    run_idx: int,
    mode: str,
    prompt_axis: int,
    prompt_idx: int,
    selected_roi: int,
    shape_original: tuple,
    dataset_name: str,
    modality: str,
    p_unet_model_name: str,
    # Prompt-UNet metrics
    p_unet_vol_dice: float,
    p_unet_window_dice: float,
    normalization_mode: str,
    num_slices_evaluated: int,
    # IFL-specific (None when mode == 'ssf')
    num_user_interacts: Optional[int],
    user_interacts_idx: Optional[List[int]],
    # nnInteractive metrics
    nn_vol_dice: float,
    nn_window_dice: float,
    # Timing
    p_unet_time_s: Optional[float] = None,
    nn_time_s: Optional[float] = None,
) -> dict:
    """Build a flat dict for one (volume, run) pair."""
    return {
        # Identification
        "volume_id"            : volume_id,
        "pid"                  : pid,
        "run_idx"              : run_idx,
        "mode"                 : mode,
        # Prompt info
        "prompt_axis"          : prompt_axis,
        "prompt_idx"           : prompt_idx,
        "selected_roi"         : int(selected_roi),
        "shape_original"       : list(shape_original),
        "dataset_name"         : dataset_name,
        "modality"             : modality,
        "p_unet_model"         : p_unet_model_name,
        "num_slices_evaluated" : num_slices_evaluated,
        # Normalization
        "normalization_mode"   : normalization_mode,
        # Prompt-UNet
        "p_unet_vol_dice"      : p_unet_vol_dice,
        "p_unet_window_dice"   : p_unet_window_dice,
        "num_user_interacts"   : num_user_interacts,   # None for SSF-only
        "user_interacts_idx"   : user_interacts_idx,   # None for SSF-only
        # nnInteractive
        "nn_vol_dice"          : nn_vol_dice,
        "nn_window_dice"       : nn_window_dice,
        # Timing
        "p_unet_time_s"        : p_unet_time_s,
        "nn_time_s"            : nn_time_s,
    }


def _compute_summary(records: List[dict]) -> dict:
    """Aggregate per-run records into human-readable summary statistics."""
    def _stats(values):
        v = [x for x in values if x is not None]
        if not v:
            return {}
        return {
            "mean"   : float(np.mean(v)),
            "std"    : float(np.std(v)),
            "median" : float(np.median(v)),
            "min"    : float(np.min(v)),
            "max"    : float(np.max(v)),
            "n"      : len(v),
        }

    p_vol    = [r["p_unet_vol_dice"]    for r in records]
    p_win    = [r["p_unet_window_dice"] for r in records]
    nn_vol   = [r["nn_vol_dice"]        for r in records]
    nn_win   = [r["nn_window_dice"]     for r in records]
    interacts= [r["num_user_interacts"] for r in records if r["num_user_interacts"] is not None]

    summary = {
        "n_runs"              : len(records),
        "mode"                : records[0]["mode"] if records else "unknown",
        "p_unet_vol_dice"     : _stats(p_vol),
        "p_unet_window_dice"  : _stats(p_win),
        "nn_vol_dice"         : _stats(nn_vol),
        "nn_window_dice"      : _stats(nn_win),
        "num_user_interacts"  : _stats(interacts) if interacts else None,
    }

    # Per-volume breakdown
    volume_ids = sorted(set(r["volume_id"] for r in records))
    per_volume = {}
    for vid in volume_ids:
        v_recs = [r for r in records if r["volume_id"] == vid]
        per_volume[vid] = {
            "n_runs"            : len(v_recs),
            "p_unet_vol_dice"   : _stats([r["p_unet_vol_dice"]    for r in v_recs]),
            "p_unet_window_dice": _stats([r["p_unet_window_dice"] for r in v_recs]),
            "nn_vol_dice"       : _stats([r["nn_vol_dice"]        for r in v_recs]),
            "nn_window_dice"    : _stats([r["nn_window_dice"]     for r in v_recs]),
        }
    summary["per_volume"] = per_volume

    return summary


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def run_benchmark(
    npz_paths: List[str],
    p_unet_model: str,
    nn_model_dir: Optional[str] = None,
    runs_per_vol: int = 5,
    mode: str = "ifl",
    modality: Optional[str] = None,
    output_threshold: float = 0.5,
    ssf_strategy: Optional[BaseSSFStrategy] = None,
    buffer_size: int = 4,
    gt_dice_threshold: float = 0.65,
    window: int = 10,
    min_prompt_pixels: int = 50,
    max_volumes: Optional[int] = None,
    return_predictions: bool = False,
    output_dir: Optional[str] = None,
    nn_device: Optional[str] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Run the full 3-D benchmark.

    Parameters
    ----------
    npz_paths       : list of str — paths to .npz datasets (ds_handler format).
    p_unet_model    : str — path to .keras model file.
    nn_model_dir    : str or None — nnInteractive weights dir.  If None, weights
                      are downloaded from HuggingFace into /tmp.
    runs_per_vol    : int — number of random prompts per volume.
    mode            : 'ssf', 'ifl' or 'ssf,ifl'
                      'ssf' — Self-Supervised Feedback only (no GT needed for driving predictions)
                      'ifl' — Interactive Feedback Loop
                      'ssf,ifl' - Both enabled
    modality        : 'CT' or 'MRI' — used only when normalization auto-detects >= v292.
    output_threshold: float — sigmoid threshold for Prompt-UNet outputs.
    ssf_strategy    : BaseSSFStrategy or None — SSF trigger strategy.  Only used when
                      mode='ssf'.  None disables SSF entirely.
    buffer_size     : int — number of recent predictions kept in the SSF buffer.
    gt_dice_threshold : float — Dice threshold for IFL GT substitution.
    window          : int — half-width for windowed Dice evaluation.
    min_prompt_pixels: int — minimum foreground pixels for prompt eligibility.
    output_dir      : str or None — directory to save pkl + json results.
                      If None, results are not saved to disk.
    nn_device       : str — device spec passed to NNInteractiveInference.
    verbose         : bool — print progress.

    Returns
    -------
    list of per-run record dicts (also saved as pkl if output_dir is set).
    """
    # --- Parse Mode ---
    mode_lower = mode.lower()
    use_ifl = "ifl" in mode_lower
    use_ssf = "ssf" in mode_lower
    active_ssf = ssf_strategy if use_ssf else None

    # --- Load Prompt-UNet ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"Loading Prompt-UNet: {p_unet_model} (IFL: {use_ifl}, SSF: {use_ssf})")

    if use_ifl:
        p_unet = InteractiveFeedbackLoop(
            model_path        = p_unet_model,
            modality          = modality,
            output_threshold  = output_threshold,
            ssf_strategy      = active_ssf,
            buffer_size       = buffer_size,
            gt_dice_threshold = gt_dice_threshold,
        )
    else:
        p_unet = VolumeInference(
            model_path       = p_unet_model,
            modality         = modality,
            output_threshold = output_threshold,
            ssf_strategy     = active_ssf,
            buffer_size      = buffer_size,
        )

    # --- Load nnInteractive ---
    if verbose:
        print(f"\nLoading nnInteractive …")
    nn_infer = NNInteractiveInference(
        model_dir = nn_model_dir,
        device    = nn_device,
    )

    # --- Iterate datasets ---
    all_records: List[dict] = []
    volume_counter = 0

    for npz_path in npz_paths:
        npz_path = str((_PROJECT_ROOT / npz_path).resolve()) if not Path(npz_path).is_absolute() else npz_path
        if verbose:
            print(f"\n{'='*60}")
            print(f"Loading dataset: {npz_path}")

        try:
            dataset = load_dataset(npz_path)
        except Exception as e:
            print(f"  ERROR loading {npz_path}: {e} — skipping.")
            continue

        dataset_name = Path(npz_path).stem
        
        # Get list of pids and optionally limit/shuffle
        pids = list(dataset.keys())
        if max_volumes is not None:
            import random
            rng = random.Random(42) # Deterministic shuffle
            rng.shuffle(pids)
            # We only want to evaluate up to max_volumes TOTAL, across all datasets
            remaining = max_volumes - volume_counter
            if remaining <= 0:
                break
            pids = pids[:remaining]

        for pid in pids:
            item = dataset[pid]
            img_3d   = np.asarray(item["image"]).astype(np.float32)
            segs     = item["segmentations"]
            vol_mod  = item.get("modality", modality)

            if vol_mod is None:
                raise ValueError(
                    f"Dataset {dataset_name} (pid {pid}) does not contain a 'modality' parameter, "
                    f"and no fallback --modality was provided."
                )

            # --- Combine all segmentation channels into one integer label map ---
            if isinstance(segs, list):
                if len(segs) == 0:
                    if verbose:
                        print(f"  [{pid}] No segmentations — skipping.")
                    continue
                seg_3d_labels = np.zeros_like(img_3d, dtype=np.int32)
                for label_idx, seg_arr in enumerate(segs, start=1):
                    seg_3d_labels[np.asarray(seg_arr) != 0] = label_idx
            else:
                seg_3d_labels = np.asarray(segs).astype(np.int32)

            if np.all(seg_3d_labels == 0):
                if verbose:
                    print(f"  [{pid}] Empty segmentation — skipping.")
                continue

            volume_id = f"{dataset_name}__{pid}"
            volume_counter += 1

            if verbose:
                print(f"\n  Volume {volume_counter}: {volume_id} | shape={img_3d.shape} | modality={vol_mod}")

            # nnInteractive needs (1, X, Y, Z)
            img_4d = np.expand_dims(img_3d, axis=0)

            for run_idx in range(runs_per_vol):
                if verbose:
                    print(f"    Run {run_idx+1}/{runs_per_vol} …", end=" ", flush=True)

                try:
                    # --- Generate random initial prompt ---
                    initial_prompt_3d, initial_prompt_2d_seg, (prompt_axis, prompt_idx), selected_roi = \
                        generate_initial_prompt(seg_3d_labels, min_pixels=min_prompt_pixels)

                    # Binary ground-truth for the selected ROI
                    seg_3d_binary = (seg_3d_labels == selected_roi).astype(np.float32)

                    # --- Prompt-UNet inference ---
                    _t0 = time.perf_counter()
                    result = p_unet.run(
                        img_3d               = img_3d,
                        seg_3d_binary        = seg_3d_binary,
                        initial_prompt_2d_seg= initial_prompt_2d_seg,
                        prompt_axis          = prompt_axis,
                        prompt_idx           = prompt_idx,
                        modality             = vol_mod,   # per-volume modality
                    )
                    p_unet_time_s = time.perf_counter() - _t0

                    user_interacts_idx = result.user_interacts_idx or []

                    # Reconstruct the full 3-D prediction volume so that
                    # P-UNet volumetric Dice is computed on the same domain
                    # as nnInteractive (full volume, not just evaluated slices).
                    p_pred_vol    = _reconstruct_volume(result, img_3d.shape)
                    p_vol_dice    = volumetric_dice(seg_3d_binary, p_pred_vol)
                    p_window_dice = dice_window_prompt(
                        result.gt_3d, result.results_3d, result.forward_indices, window=window
                    )

                    # --- nnInteractive (stdout/stderr suppressed — nnInteractive
                    #     prints hardcoded progress lines that are not controlled
                    #     by its verbose flag) ---
                    _nn_stdout = io.StringIO()
                    _nn_stderr = io.StringIO()
                    _t0 = time.perf_counter()
                    with contextlib.redirect_stdout(_nn_stdout), \
                         contextlib.redirect_stderr(_nn_stderr):
                        nn_out = nn_infer.run(
                            img_4d             = img_4d,
                            seg_3d             = seg_3d_binary,
                            initial_prompt_3d  = initial_prompt_3d,
                            user_interacts_idx = user_interacts_idx,
                            prompt_axis        = prompt_axis,
                            prompt_idx         = prompt_idx,
                            window             = window,
                        )
                    nn_time_s = time.perf_counter() - _t0

                    record = _make_run_record(
                        volume_id            = volume_id,
                        pid                  = pid,
                        run_idx              = run_idx,
                        mode                 = mode,
                        prompt_axis          = prompt_axis,
                        prompt_idx           = prompt_idx,
                        selected_roi         = selected_roi,
                        shape_original       = img_3d.shape,
                        dataset_name         = dataset_name,
                        modality             = vol_mod,
                        p_unet_model_name    = Path(p_unet_model).name,
                        normalization_mode   = result.normalization_mode,
                        num_slices_evaluated = len(result.backward_indices) + len(result.forward_indices) + 1,
                        num_user_interacts   = result.num_user_interacts,
                        user_interacts_idx   = user_interacts_idx,
                        p_unet_vol_dice      = p_vol_dice,
                        p_unet_window_dice   = p_window_dice,
                        nn_vol_dice          = nn_out["vol_dice"],
                        nn_window_dice       = nn_out["window_dice"],
                        p_unet_time_s        = p_unet_time_s,
                        nn_time_s            = nn_time_s,
                    )
                    
                    if return_predictions:
                        record["p_unet_pred_vol"] = p_pred_vol
                        record["nn_pred_vol"] = nn_out["result_volume"]
                        record["initial_prompt_3d"] = initial_prompt_3d
                        record["img_3d"] = img_3d
                        record["seg_3d_binary"] = seg_3d_binary
                    
                    all_records.append(record)

                    if verbose:
                        ui = f"  IFL={result.num_user_interacts}" if use_ifl else ""
                        print(
                            f"P-UNet vol={p_vol_dice:.3f} win={p_window_dice:.3f} ({p_unet_time_s:.1f}s)"
                            f"  |  nnInteract vol={nn_out['vol_dice']:.3f} win={nn_out['window_dice']:.3f} ({nn_time_s:.1f}s)"
                            f"{ui}"
                        )

                except Exception as e:
                    print(f"  ERROR on run {run_idx}: {e}")
                    continue

    # --- Save results ---
    if output_dir and all_records:
        out_dir = _PROJECT_ROOT / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Output dir formatting
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(p_unet_model).stem
        
        # Optional: Clean up records if return_predictions is True to avoid giant pickles
        # Actually, let's keep it as the user wants, but maybe strip out arrays before JSON dump
        
        pkl_path   = out_dir / f"results_{model_name}_{mode}_{timestamp}.pkl"
        json_path  = out_dir / f"results_{model_name}_{mode}_{timestamp}_summary.json"

        # Remove image arrays from records before pickling if return_predictions=True because
        # huge arrays will create 10+ GB pickle files on large datasets.
        # However, for single-run notebook the array return is requested in RAM, not necessarily disk.
        # I'll let pickle dump them, if it's 1-3 volumes it's fine.
        with open(pkl_path, "wb") as f:
            pickle.dump(all_records, f)
        print(f"\nFull results saved → {pkl_path}")

        summary = _compute_summary(all_records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved       → {json_path}")

        # Print a quick table to stdout
        _print_summary(summary)
    elif all_records:
        _print_summary(_compute_summary(all_records))
    else:
        print("\nNo results collected.")

    return all_records


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_summary(summary: dict):
    def _fmt(stats: Optional[dict]) -> str:
        if not stats:
            return "N/A"
        return (
            f"{stats['mean']:.3f} ± {stats['std']:.3f}  "
            f"[{stats['min']:.3f} – {stats['max']:.3f}]  "
            f"median={stats['median']:.3f}  n={stats['n']}"
        )

    print(f"\n{'='*70}")
    print(f"  BENCHMARK SUMMARY  (mode={summary['mode']}, n_runs={summary['n_runs']})")
    print(f"{'='*70}")
    print(f"  Prompt-UNet")
    print(f"    Volumetric Dice : {_fmt(summary['p_unet_vol_dice'])}")
    print(f"    Window Dice     : {_fmt(summary['p_unet_window_dice'])}")
    if summary.get("num_user_interacts"):
        print(f"    User Interactions (incl. initial): {_fmt(summary['num_user_interacts'])}")
    print(f"  nnInteractive")
    print(f"    Volumetric Dice : {_fmt(summary['nn_vol_dice'])}")
    print(f"    Window Dice     : {_fmt(summary['nn_window_dice'])}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-D benchmark: Prompt-UNet vs nnInteractive on .npz datasets."
    )
    parser.add_argument(
        "--npz_paths", nargs="+", required=True,
        help="One or more .npz file paths (ds_handler format).",
    )
    parser.add_argument(
        "--p_unet_model", required=True,
        help="Path to a saved .keras Prompt-UNet model.",
    )
    parser.add_argument(
        "--nn_model_dir", default=None,
        help="Path to nnInteractive weights directory.  If omitted, auto-downloaded.",
    )
    parser.add_argument("--runs_per_vol",      type=int,   default=5)
    parser.add_argument("--mode",              default="ifl", help="Comma-separated modes: 'ssf', 'ifl', or 'ssf,ifl'.")
    parser.add_argument("--modality",          default=None, choices=["CT", "MRI"], help="Fallback modality if not in .npz")
    parser.add_argument("--output_threshold",  type=float, default=0.5)
    parser.add_argument("--ssf_strategy",      default="none",
                        choices=["none", "relative_ssim", "mask_dice", "confidence"],
                        help="SSF trigger strategy (only used with mode=ssf).")
    parser.add_argument("--ssf_threshold",     type=float, default=0.40,
                        help="Threshold parameter for the chosen SSF strategy.")
    parser.add_argument("--buffer_size",       type=int,   default=4,
                        help="Number of recent predictions kept in the SSF buffer.")
    parser.add_argument("--gt_dice_threshold", type=float, default=0.65)
    parser.add_argument("--window",             type=int,   default=10)
    parser.add_argument("--min_prompt_pixels",  type=int,   default=50)
    parser.add_argument(
        "--output_dir",
        default="evaluation/benchmark_nninteractive/results",
        help="Directory to save pkl + json results.",
    )
    parser.add_argument("--nn_device",          default="cuda:0")

    args = parser.parse_args()

    # Build SSF strategy from CLI args
    _ssf_map = {
        "none"         : None,
        "relative_ssim": RelativeSSIMStrategy(args.ssf_threshold),
        "mask_dice"    : MaskDiceStrategy(args.ssf_threshold),
        "confidence"   : ConfidenceDropStrategy(args.ssf_threshold),
    }
    chosen_ssf = _ssf_map[args.ssf_strategy]

    run_benchmark(
        npz_paths         = args.npz_paths,
        p_unet_model      = args.p_unet_model,
        nn_model_dir      = args.nn_model_dir,
        runs_per_vol      = args.runs_per_vol,
        mode              = args.mode,
        modality          = args.modality,
        output_threshold  = args.output_threshold,
        ssf_strategy      = chosen_ssf,
        buffer_size       = args.buffer_size,
        gt_dice_threshold = args.gt_dice_threshold,
        window            = args.window,
        min_prompt_pixels = args.min_prompt_pixels,
        output_dir        = args.output_dir,
        nn_device         = args.nn_device,
    )
