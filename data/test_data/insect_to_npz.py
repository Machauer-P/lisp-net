"""
insect_to_npz.py
================
Converts the Insect micro-CT dataset (.tif) into .npz format.

Dataset structure (data/test_data/Insect/training/):
  {SpeciesName}_{idx}.tif        -- 2D micro-CT slice (520x520, uint8)
  {SpeciesName}_{idx}_mask.tif   -- Corresponding brain mask image (same shape)
                                    Brain pixels retain their intensity, background = 0.

Strategy:
  - Filter out macOS metadata files (prefixed with "._")
  - Group slices by specimen (unique species + scan number prefix, e.g. "Atta_texana")
  - Sort slices numerically by their index to form a coherent 3D stack (Z, H, W)
  - Convert mask images to binary: pixels > 0  --> 1 (brain), 0 --> background
  - Store each specimen as a 3D volume in the .npz format used by ds_handler

Label map:
  0 = Background
  1 = Brain

IMPORTANT NOTE FOR 3D EVALUATION:
  The authors independently cropped and rescaled slices from the xy, xz, and yz
  axes into separate 520x520 images. These separate axes are provided as separate
  "specimens" (e.g., Atta_texana, Atta_texana2, Atta_texana3). 
  Because they were independently rescaled, these stacks DO NOT form coherent 3D
  physical volumes! Slicing them across any axis other than axis=0 will yield
  heavily distorted and stretched images.
  Therefore, when running 3D evaluation/dataloading on this dataset, you MUST
  configure your script to ONLY slice along `axis=0`. Doing so will correctly
  evaluate the xy, xz, and yz planes natively, since each plane is encoded as
  `axis=0` in its respective specimen volume.

Note: No isotropic resampling is applied — the original paper pre-processed slices
      to 520x520 px. Voxel spacing metadata is not embedded in these TIFF files.

Reference:
  Toulkeridou et al., "Automated segmentation of insect anatomy from micro-CT
  images using deep learning", bioRxiv 2021.
"""

import os
import re
import sys
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ds_handler can be imported
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.test_data.ds_handler import save_dataset as _save_dataset_npz

# Modality tag stored in the .npz
MODALITY = "micro-CT"

# Label map (binary segmentation)
LABEL_MAP = {
    0: "Background",
    1: "Brain",
}


def _is_metadata_file(filename: str) -> bool:
    """Return True for hidden macOS metadata files (prefixed with '._')."""
    return os.path.basename(filename).startswith("._")


def _parse_specimen_and_index(filename: str):
    """
    Parse the specimen key and slice index from a TIFF filename.

    Examples
    --------
    "Atta_texana_42.tif"       -> ("Atta_texana",  42)
    "Atta_texana_42_mask.tif"  -> None  (mask files are skipped here)
    "Formica_rufa_-5.tif"      -> ("Formica_rufa", -5)

    Returns None if the filename doesn't match the expected pattern.
    """
    base = os.path.splitext(os.path.basename(filename))[0]  # strip .tif

    # Skip mask files — they are loaded separately
    if base.endswith("_mask"):
        return None

    # Match: everything up to the LAST _<integer> suffix
    # The integer may be negative (e.g. Formica_rufa_-5)
    m = re.match(r"^(.+?)_(-?\d+)$", base)
    if m is None:
        return None

    specimen_key = m.group(1)
    slice_idx = int(m.group(2))
    return specimen_key, slice_idx


def process_insect_dataset(
    data_dir: str,
    output_path: str,
    mask_threshold: int = 0,
) -> None:
    """
    Process the Insect micro-CT dataset into a .npz file.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the paired .tif files
        (e.g. ``data/test_data/Insect/training``).
    output_path : str
        Output filename stem (without .npz extension).
    mask_threshold : int, optional
        Pixels in the mask image with value > mask_threshold are labelled 1
        (brain). Default is 0 (any non-zero pixel = brain).
    """
    # -----------------------------------------------------------------------
    # 1. Collect all image TIFF files (no metadata, no mask files)
    # -----------------------------------------------------------------------
    all_tifs = glob.glob(os.path.join(data_dir, "*.tif"))
    all_tifs = [f for f in all_tifs if not _is_metadata_file(f)]

    # Build a dict:  specimen_key -> list of (slice_idx, img_path, mask_path)
    specimens: dict = defaultdict(list)
    missing_masks = []

    for img_path in all_tifs:
        parsed = _parse_specimen_and_index(img_path)
        if parsed is None:
            continue  # mask file or unrecognised name

        specimen_key, slice_idx = parsed

        # Derive the corresponding mask path
        base_no_ext = os.path.splitext(img_path)[0]  # e.g. "…/Atta_texana_42"
        mask_path = base_no_ext + "_mask.tif"

        if not os.path.exists(mask_path):
            missing_masks.append(img_path)
            continue

        specimens[specimen_key].append((slice_idx, img_path, mask_path))

    if missing_masks:
        print(
            f"WARNING: {len(missing_masks)} image(s) have no matching mask "
            f"and will be skipped."
        )

    if not specimens:
        print("Error: No valid image/mask pairs found. Check data_dir.")
        sys.exit(1)

    print(f"Found {len(specimens)} unique specimens across {data_dir}.")

    # -----------------------------------------------------------------------
    # 2. Build 3D volumes per specimen and populate the dataset dict
    # -----------------------------------------------------------------------
    dataset = {}
    success_count = 0

    sorted_keys = sorted(specimens.keys())
    total = len(sorted_keys)

    for i, key in enumerate(sorted_keys):
        slices = specimens[key]

        # Sort by slice index so the Z-axis is coherent
        slices.sort(key=lambda x: x[0])

        print(f"\n[{i + 1}/{total}] {key}  ({len(slices)} slices) ...")

        img_stack = []
        mask_stack = []

        try:
            import tifffile  # lazy import — keeps dependency optional at module level

            for _, img_path, mask_path in slices:
                img_arr = tifffile.imread(img_path).astype(np.float32)
                mask_arr = tifffile.imread(mask_path)

                # Convert masked image to binary label map
                binary_mask = (mask_arr > mask_threshold).astype(np.uint8)

                img_stack.append(img_arr)
                mask_stack.append(binary_mask)

        except Exception as exc:
            print(f"  ERROR loading slices for {key}: {exc}. Skipping.")
            continue

        img_volume = np.stack(img_stack, axis=0)    # shape (Z, H, W)
        seg_volume = np.stack(mask_stack, axis=0)   # shape (Z, H, W), uint8

        print(
            f"  -> Volume shape: {img_volume.shape} | "
            f"Brain voxels: {seg_volume.sum()} / {seg_volume.size}"
        )

        dataset[key] = {
            "image":         img_volume,
            "segmentations": seg_volume,
            "modality":      MODALITY,
        }
        success_count += 1

    # -----------------------------------------------------------------------
    # 3. Save
    # -----------------------------------------------------------------------
    if dataset:
        print(f"\nSaving {success_count} specimens to {output_path}.npz ...")
        _save_dataset_npz(dataset, output_path)
        print(f"\nLabel map: {LABEL_MAP}")
    else:
        print("\nError: No specimens were processed successfully!")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Insect micro-CT .tif → .npz converter"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Path to directory containing the paired .tif files "
            "(default: <script_dir>/Insect/training)"
        ),
    )
    parser.add_argument(
        "--output",
        default="Insect_microCT",
        help="Output filename stem without .npz extension (default: Insect_microCT)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help=(
            "Mask binarisation threshold: pixels > threshold are labelled "
            "'brain' (default: 0)"
        ),
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = (
        args.data_dir
        if args.data_dir is not None
        else os.path.join(script_dir, "Insect", "training")
    )

    if not os.path.isdir(data_dir_path):
        print(f"Error: Data directory not found: {data_dir_path}")
        sys.exit(1)

    process_insect_dataset(
        data_dir=data_dir_path,
        output_path=args.output,
        mask_threshold=args.threshold,
    )
