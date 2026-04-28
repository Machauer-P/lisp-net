"""
totalseg_to_npz.py
==================
TotalSegmentator MRI dataset → .npz converter.

Dataset: TotalSegmentator MRI — 298 (v1.0.0) / 616 (v2.0.0) whole-body MRI scans
         with 56 individual organ segmentations per subject.
Paper:   "TotalSegmentator MRI: Robust Sequence-independent Segmentation of
          Multiple Anatomic Structures in MRI" (arXiv 2405.19492)
Dataset: https://zenodo.org/records/14710732

Source format
-------------
TotalSeg_mri/
  meta.csv                  — per-subject metadata (age, scanner, sequence, split, …)
  s0001/
    mri.nii.gz              — the MRI volume (sequence-independent)
    segmentations/
      adrenal_gland_left.nii.gz
      adrenal_gland_right.nii.gz
      aorta.nii.gz
      … (56 structures total)
  s0002/
    …

Strategy
--------
For each subject folder:
  1. Load `mri.nii.gz` as the primary image.
  2. Discover all *.nii.gz files inside `segmentations/`.
  3. Combine them into a single integer label-map using the same
     priority-ordered approach as SegRap (largest → smallest volume,
     so smaller structures are not overwritten by large ones).
  4. Optionally crop to anatomy using MRI signal threshold.
  5. Resample to 1 mm isotropic spacing.
  6. Save the whole collection as a compressed .npz via ds_handler.

Usage
-----
    python totalseg_to_npz.py --data-dir TotalSeg_mri --output TotalSeg_mri

Optional flags
--------------
    --splits train,test      Only process subjects with these split labels
                             (from meta.csv; default: all).
    --no-crop                Disable anatomical cropping.
    --margin 15              Voxel margin around the anatomy bounding box.
"""

import os
import glob
import sys
import csv
from pathlib import Path

# Ensure project root is on sys.path so utils can be imported
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.test_data.ds_handler import save_dataset as _save_dataset_npz
from utils.resampling import resample_isotropic
from utils.cropping import crop_to_anatomy, mri_signal_threshold
from utils.itk_utils import load_medical_image, get_voxel_sizes
import numpy as np
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_nifti_robust(path: str, dtype=np.float32) -> tuple:
    """
    Load a NIfTI file, falling back to nibabel when SimpleITK refuses the file
    due to non-orthonormal direction cosines (a common floating-point issue in
    some TotalSegmentator MRI volumes).

    The nibabel path reads the raw voxel grid **without reorientation** so the
    resulting array axis order (Z, Y, X) always matches what SimpleITK would
    produce for the same file — avoiding shape mismatches between an image
    loaded via nibabel and its masks loaded via SimpleITK.

    Returns
    -------
    sitk_image  : SimpleITK.Image  (with voxel spacing set from the NIfTI header)
    arr         : np.ndarray       (Z, Y, X), dtype *dtype*
    """
    try:
        return load_medical_image(path, dtype=dtype)
    except RuntimeError as exc:
        if "orthonormal" not in str(exc).lower():
            raise  # re-raise unrelated errors

        # ---- nibabel fallback ----
        try:
            import nibabel as nib
        except ImportError:
            raise RuntimeError(
                f"Failed to load '{path}' with SimpleITK (non-orthonormal directions) "
                "and 'nibabel' is not installed. "
                "Run: pip install nibabel"
            ) from exc

        nii = nib.load(path)

        # Read the raw voxel data WITHOUT reorientation.
        # as_closest_canonical() would reorder axes and produce a shape that
        # differs from what SimpleITK returns for the same file (which reads the
        # raw voxel grid as-is).  Keeping the original voxel order ensures that
        # the image and its segmentation masks are always shape-compatible.
        arr = np.array(nii.get_fdata(), dtype=dtype)

        # NIfTI stores data in (i, j, k) = (X, Y, Z) order.
        # SimpleITK's GetArrayFromImage returns (Z, Y, X) — apply the same flip.
        arr = arr.transpose(2, 1, 0)

        # Build a surrogate SimpleITK image so downstream code
        # (get_voxel_sizes, resample_isotropic, …) works unchanged.
        hdr = nii.header
        # get_zooms() returns (i, j, k) = (x, y, z) voxel sizes — correct order.
        voxel_sizes_xyz = tuple(float(v) for v in hdr.get_zooms()[:3])
        img_sitk = sitk.GetImageFromArray(arr)
        img_sitk.SetSpacing(voxel_sizes_xyz)   # SimpleITK expects (x, y, z)
        img_sitk.SetOrigin((0.0, 0.0, 0.0))
        return img_sitk, arr

def _load_meta_csv(data_dir: str) -> dict:
    """Return a dict mapping image_id → row dict from meta.csv (if present)."""
    meta_path = os.path.join(data_dir, "meta.csv")
    meta = {}
    if not os.path.exists(meta_path):
        return meta
    with open(meta_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            sid = row.get("image_id", "").strip()
            if sid:
                meta[sid] = row
    return meta


def _filter_subjects(subject_folders: list, meta: dict, splits: list | None) -> list:
    """Keep only subjects whose meta 'split' value is in *splits* (if given)."""
    if not splits:
        return subject_folders
    keep = set()
    for sid, row in meta.items():
        if row.get("split", "").strip() in splits:
            keep.add(sid)
    return [f for f in subject_folders if os.path.basename(f) in keep]


def _build_combined_mask(seg_files: list, ref_shape: tuple) -> tuple[np.ndarray, dict]:
    """
    Combine per-structure binary masks into a single integer label map.

    Masks are applied largest → smallest (by foreground voxel count) so that
    small structures are not overwritten by large neighbours.

    Returns
    -------
    combined  : np.ndarray (uint8)
    oar_mapping : dict  {name: int_label}
    """
    combined = np.zeros(ref_shape, dtype=np.uint8)
    oar_mapping: dict[str, int] = {}
    next_id = 1

    mask_data = []
    for seg_file in seg_files:
        name = os.path.basename(seg_file).replace(".nii.gz", "")
        try:
            _, seg_array = _load_nifti_robust(seg_file, dtype=np.uint8)
        except Exception as exc:
            print(f"      -> WARNING: could not load '{name}': {exc}, skipping.")
            continue
        if seg_array.shape != ref_shape:
            print(
                f"      -> WARNING: shape mismatch for '{name}' "
                f"({seg_array.shape} vs {ref_shape}), skipping."
            )
            continue
        volume = int(np.sum(seg_array > 0))
        mask_data.append({"name": name, "array": seg_array, "volume": volume})

    # Apply largest first so smaller structures sit on top
    mask_data.sort(key=lambda x: -x["volume"])

    for md in mask_data:
        name = md["name"]
        if name not in oar_mapping:
            oar_mapping[name] = next_id
            next_id += 1
        oar_id = oar_mapping[name]
        combined[md["array"] > 0] = oar_id

    return combined, oar_mapping


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_totalseg_dataset(
    data_dir: str,
    output_filename: str,
    crop: bool = True,
    margin: int = 15,
    splits: list | None = None,
):
    """
    Convert TotalSegmentator MRI data to a unified .npz file.

    Parameters
    ----------
    data_dir        : Root directory containing s0001/, s0002/, … and meta.csv
    output_filename : Output path stem (without .npz)
    crop            : Apply anatomical cropping (recommended for MRI)
    margin          : Voxel margin for crop bounding box
    splits          : If given, only include subjects whose split tag is in this list
                      e.g. ["test"] to replicate the paper's internal test set
    """
    meta = _load_meta_csv(data_dir)

    # Collect subject folders (named s####)
    subject_folders = sorted(
        [f.path for f in os.scandir(data_dir)
         if f.is_dir() and f.name.startswith("s")]
    )

    if not subject_folders:
        print(f"Error: no subject folders found in '{data_dir}'.")
        sys.exit(1)

    subject_folders = _filter_subjects(subject_folders, meta, splits)
    print(f"Processing {len(subject_folders)} subject(s) from '{data_dir}' ...")

    dataset: dict = {}
    global_oar_mapping: dict[str, int] = {}
    next_global_id = 1
    skipped: list[str] = []

    for i, p_folder in enumerate(subject_folders):
        p_idx = os.path.basename(p_folder)
        print(f"\n[{i + 1}/{len(subject_folders)}] Processing {p_idx} ...")

        # ----- 1. Load MRI image -----
        img_path = os.path.join(p_folder, "mri.nii.gz")
        if not os.path.exists(img_path):
            print(f"  WARNING: mri.nii.gz not found for {p_idx}, skipping.")
            skipped.append(p_idx)
            continue

        try:
            img_sitk, img_array = _load_nifti_robust(img_path)
        except Exception as exc:
            print(f"  WARNING: Could not load image for {p_idx}: {exc}, skipping.")
            skipped.append(p_idx)
            continue

        # ----- 2. Discover segmentation files -----
        seg_dir = os.path.join(p_folder, "segmentations")
        seg_files = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))
        if not seg_files:
            print(f"  WARNING: No segmentation files found for {p_idx}, skipping.")
            skipped.append(p_idx)
            continue

        # ----- 3. Build combined label map (local IDs) -----
        # Use robust loader for segmentation masks too, so non-orthonormal
        # masks are handled identically to the image.
        combined_mask, local_oar_mapping = _build_combined_mask(
            seg_files, img_array.shape
        )

        # Promote local IDs to global consistent IDs
        remap = np.zeros(len(local_oar_mapping) + 1, dtype=np.uint8)
        for name, local_id in local_oar_mapping.items():
            if name not in global_oar_mapping:
                global_oar_mapping[name] = next_global_id
                next_global_id += 1
            remap[local_id] = global_oar_mapping[name]

        # Apply remap (vectorised index lookup)
        combined_mask = remap[combined_mask]

        # ----- 4. Crop to anatomy (MRI threshold) -----
        if crop:
            print(f"  Cropping to anatomy (margin={margin}) ...")
            try:
                img_array, combined_mask = crop_to_anatomy(
                    img_array, combined_mask, margin,
                    threshold_fn=mri_signal_threshold
                )
            except Exception as exc:
                print(f"  WARNING: Cropping failed ({exc}), using full volume.")

        # ----- 5. Resample to 1 mm isotropic -----
        voxel_sizes = get_voxel_sizes(img_sitk)
        print(
            f"  Resampling from spacing "
            f"{tuple(f'{v:.2f}' for v in voxel_sizes)} mm → 1.0 mm isotropic ..."
        )
        img_array = resample_isotropic(img_array, voxel_sizes, is_mask=False, is_ct=False)
        combined_mask = resample_isotropic(combined_mask, voxel_sizes, is_mask=True, is_ct=False)
        combined_mask = combined_mask.astype(np.uint8)
        print(f"  -> Resampled shape: {img_array.shape}")

        # ----- 6. Store -----
        dataset[p_idx] = {
            "image":         img_array,
            "segmentations": combined_mask,
            "modality":      "MRI",
        }

    if dataset:
        _save_dataset_npz(dataset, output_filename)
        print(f"\nGlobal OAR mapping ({len(global_oar_mapping)} structures):")
        for name, oid in sorted(global_oar_mapping.items(), key=lambda x: x[1]):
            print(f"  {oid:3d}: {name}")
        if skipped:
            print(f"\nSkipped subjects: {skipped}")
    else:
        print("\nError: no subjects were successfully processed!")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TotalSegmentator MRI .nii.gz → .npz converter"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to TotalSegmentator MRI root directory (contains s0001/, meta.csv, …)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz filename stem (without .npz extension)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_false",
        dest="crop",
        help="Disable anatomical cropping",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=15,
        help="Voxel margin for anatomy crop bounding box (default: 15)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "Comma-separated list of split tags to include, e.g. 'train,test'. "
            "Matches the 'split' column in meta.csv. Default: all subjects."
        ),
    )
    parser.set_defaults(crop=True)
    args = parser.parse_args()

    # Resolve default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir is None:
        args.data_dir = os.path.join(script_dir, "TotalSeg_mri")

    if not os.path.isdir(args.data_dir):
        print(f"Error: data directory not found: {args.data_dir}")
        sys.exit(1)

    out = args.output or "TotalSeg_mri"
    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None

    process_totalseg_dataset(
        data_dir=args.data_dir,
        output_filename=out,
        crop=args.crop,
        margin=args.margin,
        splits=splits,
    )
