"""
evaluation/benchmark_nninteractive/benchmark_3d.py
===================================================
Standalone benchmark script comparing Prompt-UNet (multiple modes) and
nnInteractive on 3-D volumes loaded from .npz files.

Multi-mode design
-----------------
When ``modes`` is a list of mode strings, **all modes are evaluated on the
exact same random initial prompt** within each volume run.  This enables a
fair within-prompt comparison (e.g. SSF vs IFL vs IFL+SSF).

A single ``InteractiveFeedbackLoop`` instance is loaded once and then
reconfigured between modes via ``set_ifl_enabled()`` / ``set_ssf_strategy()``
— no model reload is needed for multiple modes.

nnInteractive is run **once** per (volume, run) using the initial prompt
with no extra interactions, providing a single-click baseline.

Supported mode strings (case-insensitive)
-----------------------------------------
  'ssf'                     Self-Supervised Feedback only
  'ifl'                     Interactive Feedback Loop + GT correction, no SSF
  'ifl_ssf' / 'ssf_ifl'    IFL + SSF combined
  'none'                    Plain forward pass (no SSF, no IFL)

Inputs
------
* One or more .npz files in the standard ds_handler format.
* A saved Prompt-UNet .keras model.
* A directory containing nnInteractive model weights.

Outputs
-------
results_<model>_<modes>_<timestamp>.pkl          – Full per-run result list.
results_<model>_<modes>_<timestamp>_summary.json – Summary statistics.

Record structure (one dict per (volume, run))
---------------------------------------------
{
    # Identification & shared prompt info
    "volume_id", "pid", "run_idx", "dataset_name", "modality",
    "p_unet_model", "prompt_axis", "prompt_idx", "selected_roi",
    "shape_original", "modes_evaluated",

    # Per-mode Prompt-UNet results  (nested, keyed by canonical mode name)
    "per_mode": {
        "<mode>": {
            "vol_dice", "window_dice", "time_s",
            "normalization_mode", "num_slices_evaluated",
            "num_user_interacts",   # None for SSF / none modes
            "user_interacts_idx",   # []   for SSF / none modes
        },
        ...
    },

    # nnInteractive (single baseline run, initial prompt only)
    "nn_vol_dice", "nn_window_dice", "nn_time_s",
}

Usage (CLI)
-----------
    python evaluation/benchmark_nninteractive/benchmark_3d.py \\
        --npz_paths data/test_data/han_seg_ct.npz \\
        --p_unet_model training/p_unet_315.keras \\
        --modes ssf ifl ifl_ssf \\
        --runs_per_vol 5

Usage (from notebook / script)
-------------------------------
    from evaluation.benchmark_nninteractive.benchmark_3d import run_benchmark

    records = run_benchmark(
        npz_paths    = ["data/test_data/han_seg_ct.npz"],
        p_unet_model = "training/p_unet_315.keras",
        modes        = ["ssf", "ifl", "ifl_ssf"],
        runs_per_vol = 5,
        output_dir   = "evaluation/benchmark_nninteractive/results",
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
from typing import Dict, List, Optional, Union

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
# Mode parsing helpers
# ---------------------------------------------------------------------------

def _parse_mode(mode_str: str):
    """
    Parse a mode string into ``(use_ifl, use_ssf)`` flags.

    Accepted strings (case-insensitive, separators ignored):
        'ssf'                     → (False, True)
        'ifl'                     → (True,  False)
        'ifl_ssf' / 'ssf_ifl' / 'ssf,ifl'  → (True, True)
        'none'                    → (False, False)
    """
    m = mode_str.lower().replace("-", "_").replace(",", "_")
    use_ifl = "ifl" in m
    use_ssf = "ssf" in m
    return use_ifl, use_ssf


def _canonical_mode(mode_str: str) -> str:
    """Return the canonical mode name: 'ifl_ssf', 'ifl', 'ssf', or 'none'."""
    use_ifl, use_ssf = _parse_mode(mode_str)
    if use_ifl and use_ssf:
        return "ifl_ssf"
    if use_ifl:
        return "ifl"
    if use_ssf:
        return "ssf"
    return "none"


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _reconstruct_volume(
    result: RunResult,
    vol_shape: tuple,
) -> np.ndarray:
    """
    Rebuild a full 3-D binary prediction volume from a RunResult.

    Slices not visited remain zero.  Required so volumetric Dice is computed
    on the same domain as nnInteractive (full volume).
    """
    pred_vol = np.zeros(vol_shape, dtype=np.float32)

    ordered_indices = (
        list(reversed(result.backward_indices))
        + [result.prompt_idx]
        + result.forward_indices
    )

    slices_2d = result.results_3d
    if slices_2d.ndim == 2:
        slices_2d = slices_2d[np.newaxis]

    axis = result.prompt_axis

    for local_i, vol_idx in enumerate(ordered_indices):
        if local_i >= len(slices_2d):
            break
        s = slices_2d[local_i]
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
    modes_evaluated: List[str],
    prompt_axis: int,
    prompt_idx: int,
    selected_roi: int,
    shape_original: tuple,
    dataset_name: str,
    modality: str,
    p_unet_model_name: str,
    per_mode_results: Dict[str, dict],
    nn_vol_dice: float,
    nn_window_dice: float,
    nn_time_s: float,
) -> dict:
    """Build one unified record for a (volume, run) pair."""
    return {
        # Identification
        "volume_id"       : volume_id,
        "pid"             : pid,
        "run_idx"         : run_idx,
        "dataset_name"    : dataset_name,
        "modality"        : modality,
        "p_unet_model"    : p_unet_model_name,
        # Prompt — shared across ALL modes in this run
        "prompt_axis"     : prompt_axis,
        "prompt_idx"      : prompt_idx,
        "selected_roi"    : int(selected_roi),
        "shape_original"  : list(shape_original),
        "modes_evaluated" : list(modes_evaluated),
        # Per-mode Prompt-UNet results (nested, keyed by canonical mode name)
        "per_mode"        : per_mode_results,
        # nnInteractive — single baseline (initial prompt only)
        "nn_vol_dice"     : nn_vol_dice,
        "nn_window_dice"  : nn_window_dice,
        "nn_time_s"       : nn_time_s,
    }


def _compute_summary(records: List[dict]) -> dict:
    """Aggregate per-run records into summary statistics for all modes."""

    def _stats(values):
        v = [x for x in values if x is not None]
        if not v:
            return {}
        return {
            "mean"  : float(np.mean(v)),
            "std"   : float(np.std(v)),
            "median": float(np.median(v)),
            "min"   : float(np.min(v)),
            "max"   : float(np.max(v)),
            "n"     : len(v),
        }

    if not records:
        return {}

    modes = records[0].get("modes_evaluated", [])

    summary: dict = {
        "n_runs"         : len(records),
        "modes_evaluated": modes,
        "per_mode"       : {},
        "nn"             : {
            "vol_dice"   : _stats([r["nn_vol_dice"]    for r in records]),
            "window_dice": _stats([r["nn_window_dice"]  for r in records]),
            "time_s"     : _stats([r["nn_time_s"]       for r in records]),
        },
    }

    for mode in modes:
        mode_recs = [
            r["per_mode"][mode]
            for r in records
            if mode in r.get("per_mode", {})
        ]
        interacts = [
            m["num_user_interacts"]
            for m in mode_recs
            if m.get("num_user_interacts") is not None
        ]
        summary["per_mode"][mode] = {
            "vol_dice"          : _stats([m["vol_dice"]    for m in mode_recs]),
            "window_dice"       : _stats([m["window_dice"]  for m in mode_recs]),
            "time_s"            : _stats([m["time_s"]       for m in mode_recs]),
            "num_user_interacts": _stats(interacts) if interacts else None,
        }

    # Per-dataset breakdown
    dataset_names = sorted(set(r["dataset_name"] for r in records))
    per_dataset: dict = {}
    for ds in dataset_names:
        ds_recs = [r for r in records if r["dataset_name"] == ds]
        ds_entry: dict = {
            "n_runs"  : len(ds_recs),
            "per_mode": {},
            "nn"      : {
                "vol_dice"   : _stats([r["nn_vol_dice"]    for r in ds_recs]),
                "window_dice": _stats([r["nn_window_dice"]  for r in ds_recs]),
                "time_s"     : _stats([r["nn_time_s"]       for r in ds_recs]),
            },
        }
        for mode in modes:
            mode_recs = [
                r["per_mode"][mode]
                for r in ds_recs
                if mode in r.get("per_mode", {})
            ]
            ds_entry["per_mode"][mode] = {
                "vol_dice"   : _stats([m["vol_dice"]    for m in mode_recs]),
                "window_dice": _stats([m["window_dice"]  for m in mode_recs]),
                "time_s"     : _stats([m["time_s"]       for m in mode_recs]),
            }
        per_dataset[ds] = ds_entry
    summary["per_dataset"] = per_dataset

    # Per-volume breakdown
    volume_ids = sorted(set(r["volume_id"] for r in records))
    per_volume: dict = {}
    for vid in volume_ids:
        v_recs = [r for r in records if r["volume_id"] == vid]
        v_entry: dict = {
            "n_runs"  : len(v_recs),
            "per_mode": {},
            "nn"      : {
                "vol_dice"   : _stats([r["nn_vol_dice"]    for r in v_recs]),
                "window_dice": _stats([r["nn_window_dice"]  for r in v_recs]),
            },
        }
        for mode in modes:
            mode_recs = [
                r["per_mode"][mode]
                for r in v_recs
                if mode in r.get("per_mode", {})
            ]
            v_entry["per_mode"][mode] = {
                "vol_dice"   : _stats([m["vol_dice"]    for m in mode_recs]),
                "window_dice": _stats([m["window_dice"]  for m in mode_recs]),
            }
        per_volume[vid] = v_entry
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
    modes: Union[str, List[str]] = "ifl",
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
    batch_size: int = 3,
) -> List[dict]:
    """
    Run the 3-D benchmark with multiple Prompt-UNet modes on the **same** prompt.

    Parameters
    ----------
    npz_paths       : list of str — paths to .npz datasets (ds_handler format).
    p_unet_model    : str — path to .keras model file.
    nn_model_dir    : str or None — nnInteractive weights dir (auto-download if None).
    runs_per_vol    : int — number of random prompts evaluated per volume.
    modes           : str or list[str]
                      One or more modes, all run on the **same** initial prompt:
                        'ssf'     SSF only
                        'ifl'     IFL only
                        'ifl_ssf' IFL + SSF
                        'none'    plain baseline
                      A bare comma-separated string is also accepted.
    modality        : 'CT' or 'MRI' — fallback when not stored in .npz.
    output_threshold: float — sigmoid threshold for binary masks.
    ssf_strategy    : BaseSSFStrategy or None — used by SSF-enabled modes.
    buffer_size     : int — SSF rolling buffer depth.
    gt_dice_threshold : float — IFL Dice threshold for GT substitution.
    window          : int — half-width for windowed Dice evaluation.
    min_prompt_pixels : int — minimum foreground pixels for prompt eligibility.
    max_volumes     : int or None — cap total evaluated volumes.
    return_predictions : bool — embed prediction arrays in records (RAM-heavy).
    output_dir      : str or None — directory for pkl + json outputs.
    nn_device       : str or None — device for nnInteractive ('cuda:0', 'cpu', …).
    verbose         : bool — print progress.
    batch_size      : int — slices per GPU forward pass.

    Returns
    -------
    list of per-run dicts (one dict per (volume, run) containing all modes).
    """
    # --- Normalise modes list ---
    if isinstance(modes, str):
        modes = [m.strip() for m in modes.replace(",", " ").split() if m.strip()]
    modes_canonical = []
    seen: set = set()
    for m in modes:
        c = _canonical_mode(m)
        if c not in seen:
            modes_canonical.append(c)
            seen.add(c)
    modes = modes_canonical

    if verbose:
        print(f"\n{'='*60}")
        print(f"Loading Prompt-UNet : {p_unet_model}")
        print(f"Modes to evaluate   : {modes}")
        print(
            "(All modes share the SAME initial prompt per run; "
            "nnInteractive runs once as baseline)"
        )

    # --- Load ONE InteractiveFeedbackLoop (reused across all modes) ----------
    # set_ifl_enabled(False) makes it behave like plain VolumeInference;
    # set_ssf_strategy(None) disables SSF — so one instance covers all modes.
    p_unet = InteractiveFeedbackLoop(
        model_path        = p_unet_model,
        modality          = modality,
        output_threshold  = output_threshold,
        ssf_strategy      = ssf_strategy,   # swapped per-mode at runtime
        buffer_size       = buffer_size,
        batch_size        = batch_size,
        gt_dice_threshold = gt_dice_threshold,
    )

    # --- Load nnInteractive ---
    if verbose:
        print(f"\nLoading nnInteractive …")
    nn_infer = NNInteractiveInference(
        model_dir = nn_model_dir,
        device    = nn_device,
    )

    all_records: List[dict] = []
    volume_counter = 0

    for npz_path in npz_paths:
        npz_path = (
            str((_PROJECT_ROOT / npz_path).resolve())
            if not Path(npz_path).is_absolute()
            else npz_path
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Loading dataset: {npz_path}")

        try:
            dataset = load_dataset(npz_path)
        except Exception as e:
            print(f"  ERROR loading {npz_path}: {e} — skipping.")
            continue

        dataset_name = Path(npz_path).stem

        pids = list(dataset.keys())
        if max_volumes is not None:
            import random
            rng = random.Random(42)
            rng.shuffle(pids)
            remaining = max_volumes - volume_counter
            if remaining <= 0:
                break
            pids = pids[:remaining]

        for pid in pids:
            item    = dataset[pid]
            img_3d  = np.asarray(item["image"]).astype(np.float32)
            segs    = item["segmentations"]
            vol_mod = item.get("modality", modality)

            if vol_mod is None:
                raise ValueError(
                    f"Dataset {dataset_name} (pid {pid}) has no 'modality' field, "
                    f"and no fallback --modality was provided."
                )

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
                print(
                    f"\n  Volume {volume_counter}: {volume_id} "
                    f"| shape={img_3d.shape} | modality={vol_mod}"
                )

            img_4d = np.expand_dims(img_3d, axis=0)   # nnInteractive needs (1,X,Y,Z)

            for run_idx in range(runs_per_vol):
                if verbose:
                    print(f"    Run {run_idx+1}/{runs_per_vol}")

                try:
                    # -----------------------------------------------------------
                    # ONE random initial prompt — shared by all modes AND nnInteractive
                    # -----------------------------------------------------------
                    (
                        initial_prompt_3d,
                        initial_prompt_2d_seg,
                        (prompt_axis, prompt_idx),
                        selected_roi,
                    ) = generate_initial_prompt(seg_3d_labels, min_pixels=min_prompt_pixels)

                    seg_3d_binary = (seg_3d_labels == selected_roi).astype(np.float32)

                    # -----------------------------------------------------------
                    # Run ALL P-UNet modes on exactly the same initial prompt
                    # -----------------------------------------------------------
                    per_mode_results: Dict[str, dict] = {}

                    for mode in modes:
                        use_ifl, use_ssf = _parse_mode(mode)

                        # Reconfigure the single loaded instance
                        p_unet.set_ifl_enabled(use_ifl)
                        p_unet.set_ssf_strategy(ssf_strategy if use_ssf else None)

                        _t0 = time.perf_counter()
                        result = p_unet.run(
                            img_3d                = img_3d,
                            seg_3d_binary         = seg_3d_binary,
                            initial_prompt_2d_seg = initial_prompt_2d_seg,
                            prompt_axis           = prompt_axis,
                            prompt_idx            = prompt_idx,
                            modality              = vol_mod,
                        )
                        mode_time_s = time.perf_counter() - _t0

                        p_pred_vol    = _reconstruct_volume(result, img_3d.shape)
                        p_vol_dice    = volumetric_dice(seg_3d_binary, p_pred_vol)
                        p_window_dice = dice_window_prompt(
                            result.gt_3d,
                            result.results_3d,
                            result.forward_indices,
                            window=window,
                        )

                        mode_entry: dict = {
                            "vol_dice"            : p_vol_dice,
                            "window_dice"         : p_window_dice,
                            "time_s"              : mode_time_s,
                            "normalization_mode"  : result.normalization_mode,
                            "num_slices_evaluated": (
                                len(result.backward_indices)
                                + len(result.forward_indices)
                                + 1
                            ),
                            "num_user_interacts"  : result.num_user_interacts,
                            "user_interacts_idx"  : result.user_interacts_idx or [],
                        }
                        if return_predictions:
                            mode_entry["pred_vol"] = p_pred_vol

                        per_mode_results[mode] = mode_entry

                        if verbose:
                            ui_str = ""
                            if result.num_user_interacts is not None:
                                ui_str = f"  IFL-corrections={result.num_user_interacts - 1}"
                            print(
                                f"      [{mode:8s}]  vol={p_vol_dice:.3f}  "
                                f"win={p_window_dice:.3f}  ({mode_time_s:.1f}s){ui_str}"
                            )

                    # -----------------------------------------------------------
                    # nnInteractive — ONCE per run, initial prompt only
                    # -----------------------------------------------------------
                    _nn_stdout = io.StringIO()
                    _nn_stderr = io.StringIO()
                    _t0 = time.perf_counter()
                    with contextlib.redirect_stdout(_nn_stdout), \
                         contextlib.redirect_stderr(_nn_stderr):
                        nn_out = nn_infer.run(
                            img_4d             = img_4d,
                            seg_3d             = seg_3d_binary,
                            initial_prompt_3d  = initial_prompt_3d,
                            user_interacts_idx = [],    # initial prompt only
                            prompt_axis        = prompt_axis,
                            prompt_idx         = prompt_idx,
                            window             = window,
                        )
                    nn_time_s = time.perf_counter() - _t0

                    if verbose:
                        print(
                            f"      [nnInteract]  vol={nn_out['vol_dice']:.3f}  "
                            f"win={nn_out['window_dice']:.3f}  ({nn_time_s:.1f}s)"
                        )

                    record = _make_run_record(
                        volume_id         = volume_id,
                        pid               = pid,
                        run_idx           = run_idx,
                        modes_evaluated   = modes,
                        prompt_axis       = prompt_axis,
                        prompt_idx        = prompt_idx,
                        selected_roi      = selected_roi,
                        shape_original    = img_3d.shape,
                        dataset_name      = dataset_name,
                        modality          = vol_mod,
                        p_unet_model_name = Path(p_unet_model).name,
                        per_mode_results  = per_mode_results,
                        nn_vol_dice       = nn_out["vol_dice"],
                        nn_window_dice    = nn_out["window_dice"],
                        nn_time_s         = nn_time_s,
                    )

                    if return_predictions:
                        record["initial_prompt_3d"] = initial_prompt_3d
                        record["img_3d"]            = img_3d
                        record["seg_3d_binary"]     = seg_3d_binary
                        record["nn_pred_vol"]       = nn_out.get("result_volume")

                    all_records.append(record)

                except Exception as e:
                    import traceback
                    print(f"  ERROR on run {run_idx}: {e}")
                    traceback.print_exc()
                    continue

    # --- Save results ---
    if output_dir and all_records:
        out_dir = _PROJECT_ROOT / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(p_unet_model).stem
        modes_tag  = "_".join(modes)

        pkl_path  = out_dir / f"results_{model_name}_{modes_tag}_{timestamp}.pkl"
        json_path = out_dir / f"results_{model_name}_{modes_tag}_{timestamp}_summary.json"

        with open(pkl_path, "wb") as f:
            pickle.dump(all_records, f)
        print(f"\nFull results saved → {pkl_path}")

        summary = _compute_summary(all_records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved       → {json_path}")

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

    modes = summary.get("modes_evaluated", [])

    print(f"\n{'='*72}")
    print(f"  BENCHMARK SUMMARY   n_runs={summary.get('n_runs', '?')}")
    print(f"  Modes: {modes}")
    print(f"{'='*72}")

    for mode in modes:
        m_stats = summary.get("per_mode", {}).get(mode, {})
        ifl     = m_stats.get("num_user_interacts")
        print(f"\n  Prompt-UNet [{mode}]")
        print(f"    Volumetric Dice : {_fmt(m_stats.get('vol_dice'))}")
        print(f"    Window Dice     : {_fmt(m_stats.get('window_dice'))}")
        print(f"    Inference time  : {_fmt(m_stats.get('time_s'))}")
        if ifl:
            print(f"    User Interacts  : {_fmt(ifl)}")

    nn = summary.get("nn", {})
    print(f"\n  nnInteractive  (initial prompt only)")
    print(f"    Volumetric Dice : {_fmt(nn.get('vol_dice'))}")
    print(f"    Window Dice     : {_fmt(nn.get('window_dice'))}")
    print(f"    Inference time  : {_fmt(nn.get('time_s'))}")
    print(f"{'='*72}\n")

    # Per-dataset quick summary
    per_ds = summary.get("per_dataset", {})
    if per_ds:
        print(f"  Per-dataset overview")
        print(f"  {'Dataset':<30} {'Mode':<12} {'Vol Dice':>10}  {'Win Dice':>10}  {'Time':>8}")
        print(f"  {'-'*66}")
        for ds, ds_data in per_ds.items():
            for mode in modes:
                m = ds_data.get("per_mode", {}).get(mode, {})
                vd = m.get("vol_dice", {})
                wd = m.get("window_dice", {})
                ts = m.get("time_s", {})
                print(
                    f"  {ds:<30} [{mode:<10}]  "
                    f"{vd.get('mean', float('nan')):.3f}±{vd.get('std', float('nan')):.3f}  "
                    f"{wd.get('mean', float('nan')):.3f}±{wd.get('std', float('nan')):.3f}  "
                    f"{ts.get('mean', float('nan')):.1f}s"
                )
            nn_m = ds_data.get("nn", {})
            vd = nn_m.get("vol_dice", {})
            wd = nn_m.get("window_dice", {})
            ts = nn_m.get("time_s", {})
            print(
                f"  {ds:<30} [{'nnInteract':<10}]  "
                f"{vd.get('mean', float('nan')):.3f}±{vd.get('std', float('nan')):.3f}  "
                f"{wd.get('mean', float('nan')):.3f}±{wd.get('std', float('nan')):.3f}  "
                f"{ts.get('mean', float('nan')):.1f}s"
            )
            print(f"  {'-'*66}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "3-D benchmark: Prompt-UNet (multi-mode, same prompt) vs nnInteractive."
        )
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
        help="Path to nnInteractive weights directory (auto-download if omitted).",
    )
    parser.add_argument("--runs_per_vol", type=int, default=5)
    parser.add_argument(
        "--modes", nargs="+", default=["ifl"],
        help=(
            "One or more mode strings evaluated on the same initial prompt: "
            "'ssf', 'ifl', 'ifl_ssf', 'none'.  Example: --modes ssf ifl ifl_ssf"
        ),
    )
    parser.add_argument(
        "--modality", default=None, choices=["CT", "MRI"],
        help="Fallback modality if not stored in .npz.",
    )
    parser.add_argument("--output_threshold", type=float, default=0.5)
    parser.add_argument(
        "--ssf_strategy", default="relative_ssim",
        choices=["none", "relative_ssim", "mask_dice", "confidence"],
        help="SSF trigger strategy used by SSF-enabled modes.",
    )
    parser.add_argument(
        "--ssf_threshold", type=float, default=0.40,
        help="Threshold parameter for the chosen SSF strategy.",
    )
    parser.add_argument("--buffer_size",       type=int,   default=4)
    parser.add_argument("--gt_dice_threshold", type=float, default=0.65)
    parser.add_argument("--batch_size",        type=int,   default=3)
    parser.add_argument("--window",            type=int,   default=10)
    parser.add_argument("--min_prompt_pixels", type=int,   default=50)
    parser.add_argument(
        "--output_dir",
        default="evaluation/benchmark_nninteractive/results",
        help="Directory to save pkl + json results.",
    )
    parser.add_argument("--nn_device", default="cuda:0")

    args = parser.parse_args()

    _ssf_map = {
        "none"         : None,
        "relative_ssim": RelativeSSIMStrategy(args.ssf_threshold),
        "mask_dice"    : MaskDiceStrategy(args.ssf_threshold),
        "confidence"   : ConfidenceDropStrategy(args.ssf_threshold),
    }

    run_benchmark(
        npz_paths         = args.npz_paths,
        p_unet_model      = args.p_unet_model,
        nn_model_dir      = args.nn_model_dir,
        runs_per_vol      = args.runs_per_vol,
        modes             = args.modes,
        modality          = args.modality,
        output_threshold  = args.output_threshold,
        ssf_strategy      = _ssf_map[args.ssf_strategy],
        buffer_size       = args.buffer_size,
        batch_size        = args.batch_size,
        gt_dice_threshold = args.gt_dice_threshold,
        window            = args.window,
        min_prompt_pixels = args.min_prompt_pixels,
        output_dir        = args.output_dir,
        nn_device         = args.nn_device,
    )
