"""
Subprocess worker for isolated GPU memory profiling.

Each invocation runs in a **fresh Python process** so that GPU memory is clean
and no framework cross-talk (TF ↔ PyTorch allocator fragmentation) affects the
measurement.

Usage
-----
    python profile_worker.py --model punet --vol-info-json '{"dataset_name": ...}'
    python profile_worker.py --model nninteractive --vol-info-json '{"dataset_name": ...}'

Prints a single machine-parseable line to stdout::

    PEAK_MB: 1234.5
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project-root path injection (same pattern as the notebook)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# pynvml — framework-agnostic GPU polling
# ---------------------------------------------------------------------------
from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo)

nvmlInit()
_HANDLE = nvmlDeviceGetHandleByIndex(0)


def _gpu_used_mb():
    return nvmlDeviceGetMemoryInfo(_HANDLE).used / (1024 * 1024)


def _measure_gpu_peak(inference_fn, poll_interval_s: float = 0.1) -> float:
    """Run *inference_fn* while polling GPU used memory.  Returns peak MB."""
    peak = [_gpu_used_mb()]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            cur = _gpu_used_mb()
            if cur > peak[0]:
                peak[0] = cur
            time.sleep(poll_interval_s)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        inference_fn()
    finally:
        stop.set()
        t.join()
    return peak[0]


# ---------------------------------------------------------------------------
# Shared: single-volume NPZ loading (mmap — shape-only metadata, then load one)
# ---------------------------------------------------------------------------
_NPZ_PATHS = [
    _PROJECT_ROOT / 'data' / 'test_data' / 'TotalSeg_mri.npz',
    _PROJECT_ROOT / 'data' / 'test_data' / 'FLARE_2022.npz',
    _PROJECT_ROOT / 'data' / 'test_data' / 'han_seg_ct.npz',
    _PROJECT_ROOT / 'data' / 'test_data' / 'han_seg_mri.npz',
    _PROJECT_ROOT / 'data' / 'test_data' / 'SegRap2023.npz',
    _PROJECT_ROOT / 'data' / 'test_data' / 'HCCTase_ceCT.npz',
]
DS_NPZ_MAP = {p.stem: str(p) for p in _NPZ_PATHS}


def _load_single_volume(npz_path: str, pid: str) -> dict | None:
    data = np.load(npz_path, allow_pickle=False, mmap_mode='r')
    pids = data['_pids']
    for i, p in enumerate(pids):
        if str(p) == pid:
            item: dict = {
                'image': np.asarray(data[f'{i}_image']),
                'modality': str(data['_modalities'][i]) if '_modalities' in data else 'ct',
            }
            seg_count = int(data['_seg_counts'][i])
            segs = [np.asarray(data[f'{i}_seg_{j}']) for j in range(seg_count)]
            item['segmentations'] = segs[0] if seg_count == 1 else segs
            return item
    return None


# ======================================================================
# Profile: Prompt U-Net
# ======================================================================
def _profile_punet(vol_info: dict) -> float:
    import tensorflow as tf
    from inference.inference_volume import VolumeInference
    from inference.ssf import ConfidenceDropStrategy

    ds_name      = vol_info['dataset_name']
    pid          = vol_info['pid']
    axis         = int(vol_info['prompt_axis'])
    modality     = str(vol_info.get('modality', 'ct'))
    selected_roi = int(vol_info['selected_roi'])

    npz_path = DS_NPZ_MAP.get(ds_name)
    if npz_path is None:
        raise SystemExit(f'No NPZ for dataset {ds_name}')

    item = _load_single_volume(npz_path, pid)
    if item is None:
        raise SystemExit(f'pid {pid} not found in {ds_name}')

    img_3d = np.asarray(item['image']).astype(np.float32)
    segs   = item['segmentations']
    if isinstance(segs, list):
        seg_labels = np.zeros_like(img_3d, dtype=np.int32)
        for li, s in enumerate(segs, 1):
            seg_labels[np.asarray(s) != 0] = li
    else:
        seg_labels = np.asarray(segs).astype(np.int32)

    seg_3d_binary = (seg_labels == selected_roi).astype(np.float32)

    # Pick prompt slice: middle slice containing the ROI
    sum_axes = tuple(a for a in range(3) if a != axis)
    areas = seg_3d_binary.sum(axis=sum_axes)
    valid = np.where(areas > 0)[0]
    if len(valid) == 0:
        raise SystemExit(f'ROI {selected_roi} not found on any slice')
    prompt_idx = valid[len(valid) // 2]
    prompt_2d = np.take(seg_3d_binary, prompt_idx, axis=axis)

    model_path = str(_PROJECT_ROOT / 'training' / 'p_unet_332.keras')

    vi = VolumeInference(
        model_path=model_path,
        modality=modality,
        normalization='universal',
        ssf_strategy=ConfidenceDropStrategy(drop_fraction=0.05),
        buffer_size=4,
        batch_size=6,
    )

    # Warm-up
    _ = vi.run(
        img_3d=img_3d, seg_3d_binary=seg_3d_binary,
        initial_prompt_2d_seg=prompt_2d,
        prompt_axis=axis, prompt_idx=prompt_idx,
    )

    # Measurement
    baseline = _gpu_used_mb()
    peak = _measure_gpu_peak(
        lambda: vi.run(
            img_3d=img_3d, seg_3d_binary=seg_3d_binary,
            initial_prompt_2d_seg=prompt_2d,
            prompt_axis=axis, prompt_idx=prompt_idx,
        )
    )
    return peak - baseline


# ======================================================================
# Profile: nnInteractive
# ======================================================================
def _profile_nn(vol_info: dict) -> float:
    import torch
    from evaluation.benchmark_nninteractive.nninteractive_inference import (
        NNInteractiveInference,
    )

    ds_name      = vol_info['dataset_name']
    pid          = vol_info['pid']
    axis         = int(vol_info['prompt_axis'])
    selected_roi = int(vol_info['selected_roi'])

    npz_path = DS_NPZ_MAP.get(ds_name)
    if npz_path is None:
        raise SystemExit(f'No NPZ for dataset {ds_name}')

    item = _load_single_volume(npz_path, pid)
    if item is None:
        raise SystemExit(f'pid {pid} not found in {ds_name}')

    img_3d = np.asarray(item['image']).astype(np.float32)
    segs   = item['segmentations']
    if isinstance(segs, list):
        seg_labels = np.zeros_like(img_3d, dtype=np.int32)
        for li, s in enumerate(segs, 1):
            seg_labels[np.asarray(s) != 0] = li
    else:
        seg_labels = np.asarray(segs).astype(np.int32)

    seg_3d = (seg_labels == selected_roi).astype(np.int32)

    if axis == 1:
        img_3d = np.moveaxis(img_3d, 1, 0)
        seg_3d = np.moveaxis(seg_3d, 1, 0)
    elif axis == 2:
        img_3d = np.moveaxis(img_3d, 2, 0)
        seg_3d = np.moveaxis(seg_3d, 2, 0)

    img_4d = img_3d[np.newaxis]
    mid = img_3d.shape[0] // 2
    prompt_3d = np.zeros(img_3d.shape, dtype=np.int16)
    prompt_3d[mid] = seg_3d[mid]

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model_dir = Path(
        _PROJECT_ROOT / 'evaluation' / 'benchmark_models' / 'nnInteractive' / 'nnInteractive'
    )
    if not model_dir.exists():
        print(f'  Model dir not found: {model_dir} — falling back to auto-download')
        model_dir = None

    nn = NNInteractiveInference(model_dir=model_dir, device=device, verbose=False)

    def run_inference():
        nn.run(
            img_4d=img_4d, seg_3d=seg_3d,
            initial_prompt_3d=prompt_3d,
            user_interacts_idx=[], prompt_axis=0, prompt_idx=mid,
        )

    # Warm-up
    run_inference()

    # Measurement
    baseline = _gpu_used_mb()
    peak = _measure_gpu_peak(run_inference)

    nn.reset()
    del nn
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return peak - baseline


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Isolated GPU memory profiler')
    parser.add_argument('--model', required=True, choices=['punet', 'nninteractive'])
    parser.add_argument('--vol-info-json', required=True,
                        help='JSON dict with dataset_name, pid, prompt_axis, selected_roi, modality')
    args = parser.parse_args()

    vol_info = json.loads(args.vol_info_json)

    if args.model == 'punet':
        peak_mb = _profile_punet(vol_info)
    else:
        peak_mb = _profile_nn(vol_info)

    # Single machine-parseable output line
    print(f'PEAK_MB: {peak_mb:.1f}')


if __name__ == '__main__':
    main()
