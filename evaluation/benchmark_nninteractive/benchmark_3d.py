"""
evaluation/benchmark_nninteractive/benchmark_3d.py
===================================================
Standalone benchmark script comparing Prompt U-Net (multiple modes) and
nnInteractive on 3-D volumes loaded from .npz files.

Multi-mode / fair-comparison design
------------------------------------
All P-UNet modes in ``modes`` are evaluated on the **exact same** random
initial prompt within every (volume, run).  A single loaded
``InteractiveFeedbackLoop`` instance is reconfigured between modes via
``set_ifl_enabled()`` / ``set_ssf_strategy()`` — no model reload needed.

nnInteractive runs
------------------
nnInteractive is executed:
  • once as a **baseline** — initial prompt only (no extra interactions)
  • once **paired** with every IFL-enabled P-UNet mode, using the
    *identical* ``user_interacts_idx`` that the IFL loop produced

Example: modes = ['ssf', 'ifl', 'ifl_ssf']
  P-UNet ssf    → compared against nn_baseline   (0 extra slices)
  P-UNet ifl    → compared against nn_ifl         (same n slices as IFL)
  P-UNet ifl_ssf→ compared against nn_ifl_ssf     (same n slices as IFL+SSF)

Supported mode strings (case-insensitive)
------------------------------------------
  'ssf'                   Self-Supervised Feedback only, no GT correction
  'ifl'                   Interactive Feedback Loop (GT correction), no SSF
  'ifl_ssf' / 'ssf_ifl'  IFL + SSF combined
  'none'                  Plain forward pass (no SSF, no IFL)

Record structure (one dict per (volume, run))
----------------------------------------------
{
    # Identification & shared prompt info
    "volume_id", "pid", "run_idx", "dataset_name", "modality",
    "p_unet_model", "prompt_axis", "prompt_idx", "selected_roi",
    "shape_original", "modes_evaluated",

    # per-mode Prompt-UNet results (nested dict, keyed by canonical mode name)
    "per_mode": {
        "<mode>": {
            "vol_dice", "window_dice", "time_s",
            "normalization_mode", "num_slices_evaluated",
            "num_user_interacts",   # None for ssf / none modes
            "user_interacts_idx",   # []   for ssf / none modes
        },
        ...
    },

    # nnInteractive results (nested dict)
    "nn_results": {
        "baseline": {               # always present — initial prompt only
            "vol_dice", "window_dice", "time_s", "num_interactions": 0
        },
        "<ifl_mode>": {             # present for each IFL P-UNet mode
            "vol_dice", "window_dice", "time_s",
            "num_interactions": N   # same N as the paired P-UNet IFL mode
        },
        ...
    },
}
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
        'ssf'                         → (False, True)
        'ifl'                         → (True,  False)
        'ifl_ssf' / 'ssf_ifl' / 'ssf,ifl' → (True, True)
        'none'                        → (False, False)
    """
    m = mode_str.lower().replace("-", "_").replace(",", "_")
    return "ifl" in m, "ssf" in m


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
# nnInteractive interaction schedule
# ---------------------------------------------------------------------------

def _nn_schedule(modes: List[str], per_mode_results: Dict[str, dict]) -> List[tuple]:
    """
    Build the ordered list of ``(nn_key, user_interacts_idx)`` for all
    nnInteractive invocations in a single (volume, run).

    Returns
    -------
    list of (nn_key: str, user_interacts_idx: list[int])
        "baseline"  → []  always first
        "<mode>"    → user_interacts_idx from that IFL mode's P-UNet result
    """
    schedule = [("baseline", [])]
    for mode in modes:
        use_ifl, _ = _parse_mode(mode)
        if use_ifl:
            ui_idx = per_mode_results[mode].get("user_interacts_idx", [])
            schedule.append((mode, list(ui_idx)))
    return schedule


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _reconstruct_volume(result: RunResult, vol_shape: tuple) -> np.ndarray:
    """Rebuild a full 3-D binary prediction volume from a RunResult."""
    pred_vol = np.zeros(vol_shape, dtype=np.float32)
    ordered  = (
        list(reversed(result.backward_indices))
        + [result.prompt_idx]
        + result.forward_indices
    )
    slices_2d = result.results_3d
    if slices_2d.ndim == 2:
        slices_2d = slices_2d[np.newaxis]
    axis = result.prompt_axis
    for local_i, vol_idx in enumerate(ordered):
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
    nn_results: Dict[str, dict],
) -> dict:
    """Build one unified record for a (volume, run) pair.

    nn_results keys
    ---------------
    "baseline"   — nnInteractive with initial prompt only (no IFL).
    "<mode>"     — nnInteractive paired with that IFL P-UNet mode.
    """
    return {
        "volume_id"       : volume_id,
        "pid"             : pid,
        "run_idx"         : run_idx,
        "dataset_name"    : dataset_name,
        "modality"        : modality,
        "p_unet_model"    : p_unet_model_name,
        "prompt_axis"     : prompt_axis,
        "prompt_idx"      : prompt_idx,
        "selected_roi"    : int(selected_roi),
        "shape_original"  : list(shape_original),
        "modes_evaluated" : list(modes_evaluated),
        "per_mode"        : per_mode_results,
        "nn_results"      : nn_results,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

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

    modes    = records[0].get("modes_evaluated", [])
    nn_keys  = sorted({k for r in records for k in r.get("nn_results", {})})

    # ---------- overall ------------------------------------------------------
    summary: dict = {
        "n_runs"         : len(records),
        "modes_evaluated": modes,
        "per_mode"       : {},
        "nn_results"     : {},
    }

    for mode in modes:
        m_recs = [r["per_mode"][mode] for r in records if mode in r.get("per_mode", {})]
        ints   = [m["num_user_interacts"] for m in m_recs if m.get("num_user_interacts") is not None]
        summary["per_mode"][mode] = {
            "vol_dice"          : _stats([m["vol_dice"]    for m in m_recs]),
            "window_dice"       : _stats([m["window_dice"]  for m in m_recs]),
            "time_s"            : _stats([m["time_s"]       for m in m_recs]),
            "num_user_interacts": _stats(ints) if ints else None,
        }

    for nn_key in nn_keys:
        nn_recs = [r["nn_results"][nn_key] for r in records if nn_key in r.get("nn_results", {})]
        summary["nn_results"][nn_key] = {
            "vol_dice"        : _stats([m["vol_dice"]         for m in nn_recs]),
            "window_dice"     : _stats([m["window_dice"]       for m in nn_recs]),
            "time_s"          : _stats([m["time_s"]            for m in nn_recs]),
            "num_interactions": _stats([m["num_interactions"]  for m in nn_recs]),
        }

    # ---------- per-dataset --------------------------------------------------
    datasets    = sorted(set(r["dataset_name"] for r in records))
    per_dataset: dict = {}
    for ds in datasets:
        ds_recs = [r for r in records if r["dataset_name"] == ds]
        ds_entry: dict = {"n_runs": len(ds_recs), "per_mode": {}, "nn_results": {}}
        for mode in modes:
            m_recs = [r["per_mode"][mode] for r in ds_recs if mode in r.get("per_mode", {})]
            ds_entry["per_mode"][mode] = {
                "vol_dice"   : _stats([m["vol_dice"]    for m in m_recs]),
                "window_dice": _stats([m["window_dice"]  for m in m_recs]),
                "time_s"     : _stats([m["time_s"]       for m in m_recs]),
            }
        for nn_key in nn_keys:
            nn_recs = [r["nn_results"][nn_key] for r in ds_recs if nn_key in r.get("nn_results", {})]
            ds_entry["nn_results"][nn_key] = {
                "vol_dice"   : _stats([m["vol_dice"]    for m in nn_recs]),
                "window_dice": _stats([m["window_dice"]  for m in nn_recs]),
                "time_s"     : _stats([m["time_s"]       for m in nn_recs]),
            }
        per_dataset[ds] = ds_entry
    summary["per_dataset"] = per_dataset

    # ---------- per-volume ---------------------------------------------------
    volume_ids  = sorted(set(r["volume_id"] for r in records))
    per_volume: dict = {}
    for vid in volume_ids:
        v_recs = [r for r in records if r["volume_id"] == vid]
        v_entry: dict = {"n_runs": len(v_recs), "per_mode": {}, "nn_results": {}}
        for mode in modes:
            m_recs = [r["per_mode"][mode] for r in v_recs if mode in r.get("per_mode", {})]
            v_entry["per_mode"][mode] = {
                "vol_dice"   : _stats([m["vol_dice"]    for m in m_recs]),
                "window_dice": _stats([m["window_dice"]  for m in m_recs]),
            }
        for nn_key in nn_keys:
            nn_recs = [r["nn_results"][nn_key] for r in v_recs if nn_key in r.get("nn_results", {})]
            v_entry["nn_results"][nn_key] = {
                "vol_dice"   : _stats([m["vol_dice"]    for m in nn_recs]),
                "window_dice": _stats([m["window_dice"]  for m in nn_recs]),
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
    Run the 3-D benchmark with multiple P U-Net modes on the **same** prompt
    and paired nnInteractive runs.

    Parameters
    ----------
    npz_paths       : list[str] — .npz dataset paths (ds_handler format).
    p_unet_model    : str — path to .keras model.
    nn_model_dir    : str or None — nnInteractive weights dir (auto-download if None).
    runs_per_vol    : int — random prompts per volume.
    modes           : str or list[str]
                      All modes run on the **same** initial prompt per run.
                      Accepted: 'ssf', 'ifl', 'ifl_ssf', 'none'.
    modality        : 'CT' or 'MRI' — fallback (required if .npz has no modality).
    output_threshold: float — sigmoid→binary threshold.
    ssf_strategy    : BaseSSFStrategy or None — used by SSF-enabled modes.
    buffer_size     : int — SSF rolling buffer depth.
    gt_dice_threshold : float — IFL Dice threshold for GT substitution.
    window          : int — half-width for windowed Dice.
    min_prompt_pixels : int — minimum foreground pixels for a valid prompt.
    max_volumes     : int or None — cap on total evaluated volumes.
    return_predictions : bool — embed arrays in records (high RAM).
    output_dir      : str or None — dir for pkl + json outputs.
    nn_device       : str or None — device for nnInteractive.
    verbose         : bool — print per-run progress.
    batch_size      : int — slices per GPU forward pass.

    Returns
    -------
    list[dict] — one record per (volume, run), containing per-mode P-UNet
    results and nn_results (baseline + one per IFL mode).
    """
    # --- Normalise modes list ------------------------------------------------
    if isinstance(modes, str):
        modes = [m.strip() for m in modes.replace(",", " ").split() if m.strip()]
    seen: set = set()
    modes_c: List[str] = []
    for m in modes:
        c = _canonical_mode(m)
        if c not in seen:
            modes_c.append(c)
            seen.add(c)
    modes = modes_c

    ifl_modes = [m for m in modes if _parse_mode(m)[0]]

    if verbose:
        print(f"\n{'='*64}")
        print(f"Loading Prompt U-Net : {p_unet_model}")
        print(f"P U-Net modes        : {modes}")
        print(f"nnInteractive runs  : baseline + {ifl_modes}  (paired with IFL modes)")

    # --- Load ONE InteractiveFeedbackLoop instance ----------------------------
    p_unet = InteractiveFeedbackLoop(
        model_path        = p_unet_model,
        modality          = modality,
        output_threshold  = output_threshold,
        ssf_strategy      = ssf_strategy,
        buffer_size       = buffer_size,
        batch_size        = batch_size,
        gt_dice_threshold = gt_dice_threshold,
    )

    # --- Load nnInteractive ---------------------------------------------------
    if verbose:
        print("\nLoading nnInteractive …")
    nn_infer = NNInteractiveInference(model_dir=nn_model_dir, device=nn_device)

    all_records: List[dict] = []
    volume_counter = 0

    for npz_path in npz_paths:
        npz_path = (
            str((_PROJECT_ROOT / npz_path).resolve())
            if not Path(npz_path).is_absolute()
            else npz_path
        )
        if verbose:
            print(f"\n{'='*64}")
            print(f"Dataset: {npz_path}")

        try:
            dataset = load_dataset(npz_path)
        except Exception as e:
            print(f"  ERROR loading {npz_path}: {e} — skipping.")
            continue

        dataset_name = Path(npz_path).stem
        pids = list(dataset.keys())

        if max_volumes is not None:
            import random
            rng   = random.Random(42)
            rng.shuffle(pids)
            avail = max_volumes - volume_counter
            if avail <= 0:
                break
            pids = pids[:avail]

        for pid in pids:
            item    = dataset[pid]
            img_3d  = np.asarray(item["image"]).astype(np.float32)
            segs    = item["segmentations"]
            vol_mod = item.get("modality", modality)

            if vol_mod is None:
                raise ValueError(
                    f"Dataset {dataset_name} / pid {pid} has no 'modality' field "
                    f"and no fallback --modality was provided."
                )

            if isinstance(segs, list):
                if not segs:
                    if verbose:
                        print(f"  [{pid}] No segmentations — skipping.")
                    continue
                seg_3d_labels = np.zeros_like(img_3d, dtype=np.int32)
                for label_i, seg_arr in enumerate(segs, start=1):
                    seg_3d_labels[np.asarray(seg_arr) != 0] = label_i
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
                    f"\n  Vol {volume_counter}: {volume_id} "
                    f"| shape={img_3d.shape} | mod={vol_mod}"
                )

            img_4d = np.expand_dims(img_3d, axis=0)

            for run_idx in range(runs_per_vol):
                if verbose:
                    print(f"    Run {run_idx+1}/{runs_per_vol}")

                try:
                    # ----------------------------------------------------------
                    # ONE random prompt — shared by ALL P-UNet modes AND nn runs
                    # ----------------------------------------------------------
                    (
                        initial_prompt_3d,
                        initial_prompt_2d_seg,
                        (prompt_axis, prompt_idx),
                        selected_roi,
                    ) = generate_initial_prompt(seg_3d_labels, min_pixels=min_prompt_pixels)

                    seg_3d_binary = (seg_3d_labels == selected_roi).astype(np.float32)

                    # ----------------------------------------------------------
                    # P-UNet: all modes on the same prompt
                    # ----------------------------------------------------------
                    per_mode_results: Dict[str, dict] = {}

                    for mode in modes:
                        use_ifl, use_ssf = _parse_mode(mode)
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
                        mode_t = time.perf_counter() - _t0

                        p_pred_vol    = _reconstruct_volume(result, img_3d.shape)
                        p_vol_dice    = volumetric_dice(seg_3d_binary, p_pred_vol)
                        p_window_dice = dice_window_prompt(
                            result.gt_3d,
                            result.results_3d,
                            result.forward_indices,
                            window=window,
                        )

                        entry: dict = {
                            "vol_dice"            : p_vol_dice,
                            "window_dice"         : p_window_dice,
                            "time_s"              : mode_t,
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
                            entry["pred_vol"] = p_pred_vol
                        per_mode_results[mode] = entry

                        if verbose:
                            ui_str = ""
                            if result.num_user_interacts is not None:
                                ui_str = (
                                    f"  IFL-corrections="
                                    f"{result.num_user_interacts - 1}"
                                )
                            print(
                                f"      [P-UNet {mode:8s}]  "
                                f"vol={p_vol_dice:.3f}  win={p_window_dice:.3f}  "
                                f"({mode_t:.1f}s){ui_str}"
                            )

                    # ----------------------------------------------------------
                    # nnInteractive: baseline + one run per IFL-enabled P-UNet mode
                    #
                    #   nn_baseline   — initial prompt only (user_interacts_idx=[])
                    #   nn_<ifl_mode> — same user_interacts_idx as that IFL mode
                    # ----------------------------------------------------------
                    nn_results: Dict[str, dict] = {}
                    nn_schedule = _nn_schedule(modes, per_mode_results)

                    for nn_key, ui_idx in nn_schedule:
                        _nn_buf = io.StringIO()
                        _t0     = time.perf_counter()
                        with contextlib.redirect_stdout(_nn_buf), \
                             contextlib.redirect_stderr(_nn_buf):
                            nn_out = nn_infer.run(
                                img_4d             = img_4d,
                                seg_3d             = seg_3d_binary,
                                initial_prompt_3d  = initial_prompt_3d,
                                user_interacts_idx = ui_idx,
                                prompt_axis        = prompt_axis,
                                prompt_idx         = prompt_idx,
                                window             = window,
                            )
                        nn_t = time.perf_counter() - _t0
                        n_inter = len(ui_idx)

                        nn_results[nn_key] = {
                            "vol_dice"        : nn_out["vol_dice"],
                            "window_dice"     : nn_out["window_dice"],
                            "time_s"          : nn_t,
                            "num_interactions": n_inter,
                        }

                        if verbose:
                            label = (
                                "nn_baseline"
                                if nn_key == "baseline"
                                else f"nn+{nn_key}"
                            )
                            print(
                                f"      [{label:<16}]  "
                                f"vol={nn_out['vol_dice']:.3f}  "
                                f"win={nn_out['window_dice']:.3f}  "
                                f"({nn_t:.1f}s)  n_inter={n_inter}"
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
                        nn_results        = nn_results,
                    )

                    if return_predictions:
                        record["initial_prompt_3d"] = initial_prompt_3d
                        record["img_3d"]            = img_3d
                        record["seg_3d_binary"]     = seg_3d_binary

                    all_records.append(record)

                except Exception as e:
                    import traceback
                    print(f"  ERROR run {run_idx}: {e}")
                    traceback.print_exc()
                    continue

    # --- Save ----------------------------------------------------------------
    if output_dir and all_records:
        out_dir    = _PROJECT_ROOT / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(p_unet_model).stem
        modes_tag  = "_".join(modes)

        pkl_path  = out_dir / f"results_{model_name}_{modes_tag}_{timestamp}.pkl"
        json_path = out_dir / f"results_{model_name}_{modes_tag}_{timestamp}_summary.json"

        with open(pkl_path, "wb") as f:
            pickle.dump(all_records, f)
        print(f"\nFull results → {pkl_path}")

        summary = _compute_summary(all_records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary      → {json_path}")
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
    def _f(s):
        if not s:
            return "N/A"
        return (
            f"{s['mean']:.3f} ± {s['std']:.3f}  "
            f"[{s['min']:.3f}–{s['max']:.3f}]  "
            f"med={s['median']:.3f}  n={s['n']}"
        )

    modes   = summary.get("modes_evaluated", [])
    nn_keys = sorted(summary.get("nn_results", {}).keys())

    print(f"\n{'='*74}")
    print(f"  BENCHMARK SUMMARY   n_runs={summary.get('n_runs','?')}")
    print(f"  P-UNet modes : {modes}")
    print(f"  nn keys      : {nn_keys}")
    print(f"{'='*74}")

    for mode in modes:
        m = summary.get("per_mode", {}).get(mode, {})
        ifl = m.get("num_user_interacts")
        print(f"\n  P-UNet [{mode}]")
        print(f"    Vol  Dice  : {_f(m.get('vol_dice'))}")
        print(f"    Win  Dice  : {_f(m.get('window_dice'))}")
        print(f"    Time       : {_f(m.get('time_s'))}")
        if ifl:
            print(f"    IFL interacts: {_f(ifl)}")

    for nn_key in nn_keys:
        m = summary.get("nn_results", {}).get(nn_key, {})
        n_i = m.get("num_interactions", {})
        label = "nnInteractive [baseline]" if nn_key == "baseline" else f"nnInteractive [+{nn_key}]"
        print(f"\n  {label}")
        print(f"    Vol  Dice  : {_f(m.get('vol_dice'))}")
        print(f"    Win  Dice  : {_f(m.get('window_dice'))}")
        print(f"    Time       : {_f(m.get('time_s'))}")
        if n_i:
            print(f"    # interactions: {_f(n_i)}")

    # --- per-dataset quick table ---
    per_ds = summary.get("per_dataset", {})
    if per_ds:
        W = 68
        print(f"\n  {'='*W}")
        print(f"  {'Dataset':<28} {'Model':<22} {'Vol Dice':>10}  {'Win Dice':>10}  {'Time':>7}")
        print(f"  {'-'*W}")
        for ds, ddata in per_ds.items():
            for mode in modes:
                m  = ddata.get("per_mode", {}).get(mode, {})
                vd = m.get("vol_dice", {})
                wd = m.get("window_dice", {})
                ts = m.get("time_s", {})
                print(
                    f"  {ds:<28} [P-UNet {mode:<12}]  "
                    f"{vd.get('mean',float('nan')):.3f}±{vd.get('std',float('nan')):.3f}  "
                    f"{wd.get('mean',float('nan')):.3f}±{wd.get('std',float('nan')):.3f}  "
                    f"{ts.get('mean',float('nan')):.1f}s"
                )
            for nn_key in nn_keys:
                m  = ddata.get("nn_results", {}).get(nn_key, {})
                vd = m.get("vol_dice", {})
                wd = m.get("window_dice", {})
                ts = m.get("time_s", {})
                label = "nn_baseline" if nn_key == "baseline" else f"nn+{nn_key}"
                print(
                    f"  {ds:<28} [{label:<20}]  "
                    f"{vd.get('mean',float('nan')):.3f}±{vd.get('std',float('nan')):.3f}  "
                    f"{wd.get('mean',float('nan')):.3f}±{wd.get('std',float('nan')):.3f}  "
                    f"{ts.get('mean',float('nan')):.1f}s"
                )
            print(f"  {'-'*W}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="3-D benchmark: multi-mode Prompt U-Net vs nnInteractive (paired IFL)."
    )
    ap.add_argument("--npz_paths",      nargs="+", required=True)
    ap.add_argument("--p_unet_model",   required=True)
    ap.add_argument("--nn_model_dir",   default=None)
    ap.add_argument("--runs_per_vol",   type=int,   default=5)
    ap.add_argument(
        "--modes", nargs="+", default=["ifl"],
        help="Mode strings: 'ssf', 'ifl', 'ifl_ssf', 'none'.  Example: --modes ssf ifl ifl_ssf",
    )
    ap.add_argument("--modality",           default=None, choices=["CT", "MRI"])
    ap.add_argument("--output_threshold",   type=float, default=0.5)
    ap.add_argument(
        "--ssf_strategy", default="relative_ssim",
        choices=["none", "relative_ssim", "mask_dice", "confidence"],
    )
    ap.add_argument("--ssf_threshold",      type=float, default=0.40)
    ap.add_argument("--buffer_size",        type=int,   default=4)
    ap.add_argument("--gt_dice_threshold",  type=float, default=0.65)
    ap.add_argument("--batch_size",         type=int,   default=3)
    ap.add_argument("--window",             type=int,   default=10)
    ap.add_argument("--min_prompt_pixels",  type=int,   default=50)
    ap.add_argument("--output_dir",         default="evaluation/benchmark_nninteractive/results")
    ap.add_argument("--nn_device",          default="cuda:0")
    args = ap.parse_args()

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
