"""
memory_utils.py
===============
Shared utilities for GPU memory benchmark notebooks.

Used by ``p_unet_memory.ipynb`` and ``nninteractive_memory.ipynb`` to avoid
code duplication across the volume-scanning and data-loading sections.

Provides
--------
- ``index_npz``            : open all NPZ files once, return lookup dicts.
- ``get_seg_labels``       : cached multi-label seg loader (backed by the open NPZ).
- ``scan_candidates``      : find the largest ROI per patient per axis,
                             compute total_voxels, and classify by size.
- ``pick_largest``         : return the candidate with the largest total_voxels
                             for a given size bin.
- ``load_volume_for_memory``: load image + binary seg + prompt for a candidate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Global state (populated by index_npz, used by get_seg_labels / scan)
# ---------------------------------------------------------------------------
_NPZ: Dict[str, np.lib.npyio.NpzFile] = {}
_PID_IDX: Dict[Tuple[str, str], int] = {}
_PID_SHAPE: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
_PID_MODALITY: Dict[Tuple[str, str], str] = {}
_SEG_COUNT: Dict[Tuple[str, str], int] = {}
_SEG_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


# ---------------------------------------------------------------------------
# 1. Index NPZ files
# ---------------------------------------------------------------------------

def index_npz(npz_paths: List[Path]) -> None:
    """Open every NPZ in *npz_paths* (mmap) and populate the global lookup
    dicts so that subsequent calls to ``get_seg_labels`` and
    ``scan_candidates`` are fast.

    Parameters
    ----------
    npz_paths : list[Path]
        Paths to .npz test-data files.
    """
    _NPZ.clear(); _PID_IDX.clear(); _PID_SHAPE.clear()
    _PID_MODALITY.clear(); _SEG_COUNT.clear(); _SEG_CACHE.clear()

    print("Indexing NPZ files ...")
    for path in npz_paths:
        ds_name = path.stem
        data = np.load(str(path), allow_pickle=False, mmap_mode="r")
        _NPZ[ds_name] = data
        modalities = data["_modalities"] if "_modalities" in data else None
        for i, p in enumerate(data["_pids"]):
            p = str(p)
            key = (ds_name, p)
            _PID_IDX[key] = i
            _PID_SHAPE[key] = data[f"{i}_image"].shape  # (D0, D1, D2)
            _PID_MODALITY[key] = (
                str(modalities[i]) if modalities is not None else "ct"
            )
            _SEG_COUNT[key] = int(data["_seg_counts"][i])
    print(f"  {len(_PID_IDX)} patients indexed")


# ---------------------------------------------------------------------------
# 2. Cached segmentation loader
# ---------------------------------------------------------------------------

def get_seg_labels(ds_name: str, pid: str) -> Optional[np.ndarray]:
    """Return a 3-D int32 multi-label segmentation array for *pid*.

    Materialised on first access; cached thereafter.
    """
    key = (ds_name, str(pid))
    if key in _SEG_CACHE:
        return _SEG_CACHE[key]
    data = _NPZ.get(ds_name)
    if data is None:
        return None
    i = _PID_IDX.get(key)
    if i is None:
        return None
    seg_count = _SEG_COUNT.get(key, 0)
    if seg_count == 1:
        seg_labels = np.asarray(data[f"{i}_seg_0"]).astype(np.int32)
    else:
        shape = _PID_SHAPE[key]
        seg_labels = np.zeros(shape, dtype=np.int32)
        for j in range(seg_count):
            s = np.asarray(data[f"{i}_seg_{j}"])
            seg_labels[s != 0] = j + 1
    _SEG_CACHE[key] = seg_labels
    return seg_labels


# ---------------------------------------------------------------------------
# 3. Volume scanning
# ---------------------------------------------------------------------------

def scan_candidates(
    npz_paths: List[Path],
    small_limit: int = 128**3,
    large_limit: int = 192**3,
) -> List[Dict[str, Any]]:
    """Find the largest ROI per patient per axis and classify by total_voxels.

    Parameters
    ----------
    npz_paths : list[Path]
    small_limit, large_limit : int
        Classification thresholds for ``total_voxels``.

    Returns
    -------
    list[dict]
        Each dict has keys: ``npz_path``, ``dataset_name``, ``pid``,
        ``modality``, ``axis``, ``roi``, ``roi_slices``, ``h``, ``w``,
        ``total_voxels``, ``size_bin``.
    """
    candidates: List[Dict[str, Any]] = []
    print("Scanning volumes (largest ROI per patient per axis) ...")

    for path in npz_paths:
        ds_name = path.stem
        data = _NPZ[ds_name]
        for _, p in enumerate(data["_pids"]):
            p = str(p)
            seg_labels = get_seg_labels(ds_name, p)
            if seg_labels is None:
                continue
            d0, d1, d2 = _PID_SHAPE[(ds_name, p)]
            shape = (d0, d1, d2)
            modality = _PID_MODALITY.get((ds_name, p), "ct")

            counts = np.bincount(seg_labels.ravel())
            if len(counts) <= 1:
                continue
            best_roi = int(np.argmax(counts[1:]) + 1)

            for axis in range(3):
                h, w = [shape[a] for a in range(3) if a != axis]
                mask = np.moveaxis(seg_labels == best_roi, axis, 0)
                roi_slices = int(np.any(mask, axis=(1, 2)).sum())
                if roi_slices == 0:
                    continue
                total_voxels = roi_slices * h * w
                size_bin = _classify(total_voxels, small_limit, large_limit)
                candidates.append(
                    {
                        "npz_path": str(path),
                        "dataset_name": ds_name,
                        "pid": p,
                        "modality": modality,
                        "axis": axis,
                        "roi": best_roi,
                        "roi_slices": roi_slices,
                        "h": h,
                        "w": w,
                        "total_voxels": total_voxels,
                        "size_bin": size_bin,
                    }
                )

    print(
        f"Scanned {len(candidates)} (patient, axis) combinations "
        f"(largest ROI only)"
    )
    return candidates


def _classify(total_voxels: int, small_limit: int, large_limit: int) -> str:
    if total_voxels <= small_limit:
        return "Small"
    if total_voxels <= large_limit:
        return "Medium"
    return "Large"


# ---------------------------------------------------------------------------
# 4. Pick the heaviest candidate for a size bin
# ---------------------------------------------------------------------------

def pick_largest(
    candidates: List[Dict[str, Any]], size_label: str
) -> Optional[Dict[str, Any]]:
    """Return the candidate with the largest ``total_voxels`` in *size_label*."""
    subset = [c for c in candidates if c["size_bin"] == size_label]
    if not subset:
        return None
    return max(subset, key=lambda c: c["total_voxels"])


# ---------------------------------------------------------------------------
# 5. Load volume data ready for memory profiling
# ---------------------------------------------------------------------------

def load_volume_for_memory(
    candidate: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load image + binary seg + prompt for *candidate*.

    Returns
    -------
    img_3d : np.ndarray  (float32)
    seg_3d_binary : np.ndarray  (float32, 0/1)
    prompt_2d : np.ndarray  (float32, 0/1)
    prompt_idx : int
    """
    from data.test_data.ds_handler import load_dataset as _load_ds

    dataset = _load_ds(candidate["npz_path"])
    item = dataset[candidate["pid"]]
    img_3d = np.asarray(item["image"]).astype(np.float32)
    segs = item["segmentations"]

    if isinstance(segs, list):
        seg_labels = np.zeros_like(img_3d, dtype=np.int32)
        for li, s in enumerate(segs, 1):
            seg_labels[np.asarray(s) != 0] = li
    else:
        seg_labels = np.asarray(segs).astype(np.int32)

    seg_3d_binary = (seg_labels == candidate["roi"]).astype(np.float32)

    # Pick middle slice containing the ROI as the prompt
    sum_axes = tuple(a for a in range(3) if a != candidate["axis"])
    areas = seg_3d_binary.sum(axis=sum_axes)
    valid = np.where(areas > 0)[0]
    prompt_idx = valid[len(valid) // 2]
    prompt_2d = np.take(seg_3d_binary, prompt_idx, axis=candidate["axis"])

    return img_3d, seg_3d_binary, prompt_2d, prompt_idx
