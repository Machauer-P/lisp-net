"""
generate_2d_test_data_legacy.py
===============================
Legacy version of the 2D evaluation dataset generator.
This recreates the exact spatial scaling and normalization approach
used by the old DataGenerator (DataGenerator_old.py) without touching
the new codebase or requiring broken dependencies.

Legacy approach recreated here:
1. Spatial Scale: Entire 2D cross-sections are RESIZED to 128x128 
   (instead of extracting a 128x128 crop).
2. Normalization: Min-max normalization [0, 1] is applied INDEPENDENTLY 
   on each 2D slice (instead of over the entire 3D volume).

This script outputs standard NPZ bundles that can be directly evaluated
by the current `eval_pipeline_2d` without any code changes.
"""

import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom

# Resolve project root (two levels above)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.test_data.ds_handler_2d import save_2d_npz_bundle
from data.DataLoader_npz import DataLoader_npz

def legacy_min_max_norm(slice_2d, lower_q=0.5, upper_q=99.5):
    """Recreates the old per-slice Min-Max percentile normalization."""
    # If slice is completely empty or homogeneous, avoid division by zero
    if np.max(slice_2d) == np.min(slice_2d):
        return np.zeros_like(slice_2d, dtype=np.float32)
    
    q_min = np.percentile(slice_2d, lower_q)
    q_max = np.percentile(slice_2d, upper_q)
    
    # Avoid division by zero if quantiles are identical
    if q_max == q_min:
        return np.zeros_like(slice_2d, dtype=np.float32)
        
    clipped = np.clip(slice_2d, q_min, q_max)
    normed = (clipped - q_min) / (q_max - q_min + 1e-8)
    return normed.astype(np.float32)

def legacy_resize(array_2d, target_size=(128, 128), is_mask=False):
    """Recreates the old behavior of resizing the entire slice to 128x128."""
    h, w = array_2d.shape
    zoom_factors = (target_size[0] / h, target_size[1] / w)
    order = 0 if is_mask else 1  # 0 = nearest for masks, 1 = bilinear for images
    resized = zoom(array_2d, zoom_factors, order=order, mode='nearest')
    return resized

def sample_offset(i, offset, length):
    possible = list(range(-offset, 0)) + list(range(1, offset + 1))
    r = random.choice(possible)
    if (i + r < 0) or (i + r >= length):
        return None
    return r

def gen_save_ds_legacy(dl, path, ds_name, offset_val, num_ds=100, max_data_points=100, len_p_set=16, minimum_pixel=5):
    os.makedirs(path, exist_ok=True)
    
    dimensions = ['x', 'y', 'z']
    
    for i in range(num_ds):
        print(f"\\nGenerating legacy dataset {i + 1}/{num_ds}...")
        
        x_new, y_new, p_new, offset_list, m_new = [], [], [], [], []
        slices_added = 0
        task = None
        dim = None
        target_total = max_data_points + len_p_set
        
        dl.current_ids = dl.train_ids
        np.random.shuffle(dl.current_ids)
        
        # 1. Pick a random dimension and task unconditionally
        dim = random.choice(dimensions)
        dim_idx = 'xyz'.index(dim)
        
        for pid in dl.current_ids:
            y_raw = dl.dataset[pid]['segmentations']
            if isinstance(y_raw, list):
                if len(y_raw) > 0:
                    task = random.randint(1, len(y_raw))
                    break
            else:
                y_arr = np.asarray(y_raw)
                valid = [int(v) for v in np.unique(y_arr) if v != 0]
                if valid:
                    task = random.choice(valid)
                    break
        
        if task is None:
            print("No valid tasks found. Skipping.")
            continue
            
        print(f"Current task (Global): {task} along axis {dim}")

        # 2. Collect slices across patients
        np.random.shuffle(dl.current_ids)
        for pid in dl.current_ids:
            if slices_added >= target_total:
                break
                
            current_dict = dl.dataset[pid]
            vol_x = np.asarray(current_dict['image'], dtype=np.float32)
            vol_y = current_dict['segmentations']
            is_mri = 1.0 if current_dict.get('modality', 'UNKNOWN') != 'CT' else 0.0
            
            if isinstance(vol_y, list):
                if task > len(vol_y): continue
                task_vol = np.asarray(vol_y[task - 1])
                if task_vol.sum() == 0: continue
            else:
                y_arr = np.asarray(vol_y)
                if task not in np.unique(y_arr): continue
                task_vol = y_arr == task

            y_shape = task_vol.shape[dim_idx]
            slice_indices = list(range(y_shape))
            random.shuffle(slice_indices)
            
            for idx in slice_indices:
                if slices_added >= target_total:
                    break
                
                r = sample_offset(idx, offset_val, y_shape)
                if r is None:
                    continue
                
                # Extract raw 2D slices
                if dim == 'x':
                    y_slice = task_vol[idx, :, :]
                    yr_slice = task_vol[idx+r, :, :]
                    x_slice = vol_x[idx, :, :]
                    xr_slice = vol_x[idx+r, :, :]
                elif dim == 'y':
                    y_slice = task_vol[:, idx, :]
                    yr_slice = task_vol[:, idx+r, :]
                    x_slice = vol_x[:, idx, :]
                    xr_slice = vol_x[:, idx+r, :]
                else:
                    y_slice = task_vol[:, :, idx]
                    yr_slice = task_vol[:, :, idx+r]
                    x_slice = vol_x[:, :, idx]
                    xr_slice = vol_x[:, :, idx+r]
                
                # Legacy conditions: must have some minimum pixels before resizing
                if np.count_nonzero(y_slice) < minimum_pixel or np.count_nonzero(yr_slice) < minimum_pixel:
                    continue
                    
                # Apply legacy per-slice min-max normalization
                x_norm = legacy_min_max_norm(x_slice)
                xr_norm = legacy_min_max_norm(xr_slice)
                
                # Apply legacy resizing (compresses full cross-section into 128x128)
                x_res = legacy_resize(x_norm, is_mask=False)
                xr_res = legacy_resize(xr_norm, is_mask=False)
                y_res = legacy_resize(y_slice, is_mask=True)
                yr_res = legacy_resize(yr_slice, is_mask=True)
                
                # Check valid pixels again after resize
                if np.count_nonzero(y_res) < minimum_pixel or np.count_nonzero(yr_res) < minimum_pixel:
                    continue
                
                # Expand dims to shape (128, 128, 1)
                x_res = np.expand_dims(x_res, axis=-1)
                xr_res = np.expand_dims(xr_res, axis=-1)
                y_res = np.expand_dims(y_res, axis=-1).astype(np.float32)
                yr_res = np.expand_dims(yr_res, axis=-1).astype(np.float32)
                
                # Prompt representation: [xr_res | yr_res]
                p_res = np.concatenate([xr_res, yr_res], axis=-1)
                
                x_new.append(x_res)
                y_new.append(y_res)
                p_new.append(p_res)
                offset_list.append(r)
                m_new.append(is_mri)
                
                slices_added += 1

        if slices_added < len_p_set + 1:
            print(f"  WARNING: not enough samples for a query set (total={slices_added}). Skipping.")
            continue
            
        x_np = np.stack(x_new)
        y_np = np.stack(y_new)
        p_np = np.stack(p_new)
        offset_arr = np.array(offset_list, dtype=np.int32)
        m_np = np.array(m_new, dtype=np.float32)
        
        # Shuffle for randomness
        indices = np.arange(slices_added)
        np.random.shuffle(indices)
        x_np, y_np, p_np, offset_arr, m_np = x_np[indices], y_np[indices], p_np[indices], offset_arr[indices], m_np[indices]
        
        support_end = min(len_p_set, slices_added)
        query_start = support_end
        
        # Note: we provide x_np for both regular and _u inputs because in the legacy pipeline,
        # both UniverSeg and Prompt-UNet received the same 0-1 min-max normalized slices!
        support_data = {
            'sx':         x_np[:support_end],
            'sx_u':       x_np[:support_end],
            'sy':         y_np[:support_end],
            's_modality': m_np[:support_end],
        }
        query_data = {
            'x':        x_np[query_start:],
            'x_u':      x_np[query_start:],
            'y':        y_np[query_start:],
            'p':        p_np[query_start:],
            'offset':   offset_arr[query_start:],
            'modality': m_np[query_start:],
        }

        save_2d_npz_bundle(query_data, support_data, filename=f"{i}_{ds_name}", path=path)

def main():
    parser = argparse.ArgumentParser(description="Generate Legacy 2D benchmark datasets.")
    parser.add_argument("--input_npz", type=str, nargs='+', default=["data/test_data/han_seg_mri.npz"])
    parser.add_argument("--output_dir", type=str, default="data/test_data/2d/legacy_offset_10")
    parser.add_argument("--ds_name", type=str, default="hanseg")
    parser.add_argument("--num_ds", type=int, default=100)
    parser.add_argument("--offset", type=int, default=10)
    parser.add_argument("--max_points", type=int, default=100)
    parser.add_argument("--len_p_set", type=int, default=16)
    args = parser.parse_args()

    input_paths = []
    for p in args.input_npz:
        full_p = os.path.join(project_root, p) if not os.path.isabs(p) else p
        if not os.path.exists(full_p):
            print(f"Error: Could not find input file: {full_p}")
            sys.exit(1)
        input_paths.append(full_p)

    output_path = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    print(f"Initializing DataLoader from {input_paths}...")
    dl = DataLoader_npz(input_paths, val_size=0.0)

    gen_save_ds_legacy(dl, output_path, args.ds_name, args.offset, args.num_ds, args.max_points, args.len_p_set)
    print(f"\\nLegacy datasets successfully generated in {output_path}")

if __name__ == "__main__":
    main()
