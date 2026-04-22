"""
generate_2d_test_data.py
========================
Standalone script (and importable module) for generating 2D evaluation
datasets used in few-shot segmentation benchmarks (e.g., UniverSeg comparison).

Each generated file is a single NPZ bundle named  {i}_{ds_name}.npz
containing a query set and a support set for ONE anatomical structure
(picked randomly via DataGenerator.get_data_points_from_one_task_numpy).

Storage convention
------------------
  x / sx  : z-score images, clipped to [-5, 5]  (p_unet_292 training range)
  y / sy  : binary segmentation labels in {0, 1}
  p       : prompt = [reference_image | reference_label] stacked on channel dim
  offset  : signed slice offset used to build each pair
  modality: 0.0 = CT, 1.0 = MRI  — stored per sample for future multi-modal use

Usage
-----
  # via CLI:
  python generate_2d_test_data.py \\
      --input_npz data/test_data/han_seg_mri.npz data/test_data/han_seg_ct.npz \\
      --output_dir data/test_data/2d/offset_5 \\
      --ds_name hanseg --num_ds 100

  # or from a notebook using gen_save_ds() directly.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Resolve project root (two levels above this file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.test_data.ds_handler_2d import save_2d_npz_bundle
from data.DataLoader_npz import DataLoader_npz
from data.DataGenerator import DataGenerator


def gen_save_ds(dg, path, ds_name, offset_val,
                num_ds=100, max_data_points=100, len_p_set=16,
                dimensions=None):
    """Generate and save 2D evaluation datasets as NPZ bundles.

    Each bundle corresponds to a single randomly-chosen anatomical task
    (structure).  The first `len_p_set` samples become the support set;
    the remainder become the query set.

    Args:
        dg              : DataGenerator instance (already initialised).
        path            : Output directory (created if missing).
        ds_name         : Base name embedded in each output filename.
        offset_val      : Signed slice offset for prompt-target pairs.
        num_ds          : Number of bundles to generate.
        max_data_points : Total samples to generate per bundle
                          (support + query).  Actual count may be lower
                          if the task is exhausted.
        len_p_set       : Number of samples reserved for the support set.
        dimensions      : List of axes to sample from ('x', 'y', 'z').
                          None → all three axes.
    """
    os.makedirs(path, exist_ok=True)

    for i in range(num_ds):
        print(f"\nGenerating dataset {i + 1}/{num_ds}...")

        x_np, y_np, p_np, x_u_np, m_np, offset_list, task = dg.get_data_points_from_one_task_numpy(
            max_data_points=max_data_points + len_p_set,
            offset=offset_val,
            dimensions=dimensions,
            extraction_mode='fullslice',
            return_task=True
        )

        total       = len(x_np)
        support_end = min(len_p_set, total)
        query_start = support_end

        if query_start >= total:
            print(f"  WARNING: not enough samples for a query set "
                  f"(total={total}, support_end={support_end}). Skipping.")
            continue

        support_data = {
            'sx':         x_np[:support_end],     # Prompt-UNet support (z-score [-5,5])
            'sx_u':       x_u_np[:support_end],   # UniverSeg support (min-max [0,1])
            'sy':         y_np[:support_end],
            's_modality': m_np[:support_end],
        }
        query_data = {
            'x':        x_np[query_start:],
            'x_u':      x_u_np[query_start:],     # UniverSeg query (min-max [0,1])
            'y':        y_np[query_start:],
            'p':        p_np[query_start:],
            'offset':   np.array(offset_list[query_start:], dtype=np.int32),
            'modality': m_np[query_start:],
            'task':     task,
        }

        save_2d_npz_bundle(query_data, support_data,
                           filename=f"{i}_{ds_name}",
                           path=path)

    print(f"\nSuccess: Generated datasets in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D benchmark datasets (NPZ bundles)."
    )
    parser.add_argument(
        "--input_npz", type=str, nargs='+',
        default=["data/test_data/han_seg_mri.npz"],
        help="Path(s) to input .npz file(s) (relative to project root or absolute).",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/test_data/2d/offset_5",
        help="Directory to store generated NPZ bundles.",
    )
    parser.add_argument(
        "--ds_name", type=str, default="hanseg",
        help="Base name embedded in output filenames.",
    )
    parser.add_argument(
        "--num_ds", type=int, default=100,
        help="Number of bundles to generate.",
    )
    parser.add_argument(
        "--offset", type=int, default=5,
        help="Signed slice offset for prompt-target pairs.",
    )
    parser.add_argument(
        "--max_points", type=int, default=100,
        help="Target number of query samples per bundle.",
    )
    parser.add_argument(
        "--len_p_set", type=int, default=16,
        help="Number of samples reserved for the support set.",
    )
    args = parser.parse_args()

    # Resolve input paths
    input_paths = []
    for p in args.input_npz:
        full_p = os.path.join(project_root, p) if not os.path.isabs(p) else p
        if not os.path.exists(full_p) and os.path.exists(p):
            full_p = os.path.abspath(p)
        if not os.path.exists(full_p):
            print(f"Error: Could not find input file: {full_p}")
            sys.exit(1)
        input_paths.append(full_p)

    output_path = (
        os.path.join(project_root, args.output_dir)
        if not os.path.isabs(args.output_dir)
        else args.output_dir
    )

    print(f"Initializing DataLoader from {input_paths}...")
    dl = DataLoader_npz(input_paths, val_size=0.0)
    dg = DataGenerator(dl)

    gen_save_ds(
        dg=dg,
        path=output_path,
        ds_name=args.ds_name,
        offset_val=args.offset,
        num_ds=args.num_ds,
        max_data_points=args.max_points,
        len_p_set=args.len_p_set,
    )


if __name__ == "__main__":
    main()
