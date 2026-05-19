"""
couinaud_to_npz.py
==================
Converts the Couinaud Liver Segmentation dataset (NIfTI) into .npz format.

Dataset structure:
  - CT images   : ``Angelou0516/msd-hepatic-vessel`` (HuggingFace)
    → ``imagesTr/hepaticvessel_XXX.nii.gz``  (303 MSD Task08 volumes)
  - Couinaud masks : ``Angelou0516/couinaud-liver`` (HuggingFace)
    → ``hepaticvessel_XXX.nii.gz``  (161 annotated volumes)
  - Mapping     : ``train.jsonl`` in the couinaud-liver repo

The CT images are standard medical CT (HU-calibrated) from the Medical
Segmentation Decathlon Task 8 (Hepatic Vessel).  The masks contain 8 Couinaud
liver segments (I–VIII) as label values 1–8 (0 = background).

Label map
---------
    0 = Background
    1 = Caudate lobe (I)
    2 = Left lateral superior segment (II)
    3 = Left lateral inferior segment (III)
    4 = Left medial segment (IV)
    5 = Right anterior inferior segment (V)
    6 = Right posterior inferior segment (VI)
    7 = Right posterior superior segment (VII)
    8 = Right anterior superior segment (VIII)

Reference
---------
Couinaud, C. (1957). Le foie: études anatomiques et chirurgicales. Masson.
MSD Task08: http://medicaldecathlon.com
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import OrderedDict

# Suppress obnoxious HF symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

# ---------------------------------------------------------------------------
# Project-root path injection
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.test_data.ds_handler import save_dataset as _save_dataset_npz

MODALITY = "CT"

LABEL_MAP = OrderedDict([
    (0, "Background"),
    (1, "Caudate lobe (I)"),
    (2, "Left lateral superior segment (II)"),
    (3, "Left lateral inferior segment (III)"),
    (4, "Left medial segment (IV)"),
    (5, "Right anterior inferior segment (V)"),
    (6, "Right posterior inferior segment (VI)"),
    (7, "Right posterior superior segment (VII)"),
    (8, "Right anterior superior segment (VIII)"),
])


def _download_file(repo_id: str, filename: str, local_dir: str) -> str:
    """Download a single file from HuggingFace, placing it at the same relative
    path under ``local_dir``.  Returns the local filesystem path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )


def _download_jsonl(repo_id: str, filename: str, local_dir: str) -> str:
    """Download JSONL mapping file."""
    return _download_file(repo_id, filename, local_dir)


def process_couinaud_dataset(
    output_path: str,
    max_volumes: int = 40,
    cache_dir: str | None = None,
) -> None:
    """
    Download Couinaud Liver masks + MSD Hepatic Vessel CT images from
    HuggingFace, pair them, and save as a single .npz file.

    Only the *needed* files are downloaded — the JSONL mapping is fetched
    first, then exactly ``max_volumes`` mask + image pairs are pulled.

    The NPZ is saved uncompressed so it can be memory-mapped on load via
    ``load_dataset(path, mmap_mode='r')``.

    Parameters
    ----------
    output_path : str
        Output filename stem (without .npz extension).
    max_volumes : int
        Maximum number of patients / volumes to include (default 40).
    cache_dir : str or None
        Directory for HuggingFace cache.  Uses ``_couinaud_cache`` next to
        this script if None.
    """
    import nibabel as nib

    MASK_REPO = "Angelou0516/couinaud-liver"
    IMAGE_REPO = "Angelou0516/msd-hepatic-vessel"

    # -------------------------------------------------------------------
    # 1. Prepare cache
    # -------------------------------------------------------------------
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_couinaud_cache"
        )
    os.makedirs(cache_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # 2. Download JSONL mapping ONLY (cheap — text file)
    # -------------------------------------------------------------------
    print(f"  Fetching JSONL from {MASK_REPO} …")
    jsonl_local = _download_jsonl(MASK_REPO, "train.jsonl", cache_dir)

    with open(jsonl_local, encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]

    print(f"  JSONL entries: {len(entries)}")

    # -------------------------------------------------------------------
    # 3. Download only the files we need, one by one
    # -------------------------------------------------------------------
    dataset: dict = {}
    paired = 0
    missing = 0

    for entry in entries:
        if paired >= max_volumes:
            break

        pid = entry["patient_id"].removesuffix(".nii.gz")

        # Filenames are just the basename of the JSONL paths
        img_name = os.path.basename(entry["image"])  # hepaticvessel_XXX.nii.gz
        mask_name = os.path.basename(entry["mask"])   # same stem
        stem = img_name.removesuffix(".nii.gz")

        # Download mask from couinaud-liver repo
        try:
            mask_local = _download_file(MASK_REPO, mask_name, cache_dir)
        except Exception as exc:
            print(f"  WARNING: mask not found for {stem}: {exc}")
            missing += 1
            continue

        # Download CT image from msd-hepatic-vessel repo
        img_repo_path = f"imagesTr/{img_name}"
        try:
            img_local = _download_file(IMAGE_REPO, img_repo_path, cache_dir)
        except Exception as exc:
            print(f"  WARNING: image not found for {stem}: {exc}")
            missing += 1
            continue

        # Load (read directly as float32 to avoid nibabel's default float64)
        try:
            img_nii = nib.load(img_local)
            mask_nii = nib.load(mask_local)
            img_data = np.asarray(img_nii.dataobj, dtype=np.float32)
            mask_data = np.asarray(mask_nii.dataobj, dtype=np.uint8)
        except Exception as exc:
            print(f"  WARNING: Failed to load {stem}: {exc}. Skipping.")
            missing += 1
            continue

        # NIfTI stores data as (X, Y, Z) → permute to (Z, H, W)
        img_data = np.transpose(img_data, (2, 0, 1))
        mask_data = np.transpose(mask_data, (2, 0, 1))

        # Sanity: image and mask must have the same spatial shape
        if img_data.shape != mask_data.shape:
            print(f"  WARNING: shape mismatch {img_data.shape} vs {mask_data.shape}. Skipping.")
            missing += 1
            continue

        nz_labels = np.setdiff1d(np.unique(mask_data), [0])
        print(
            f"  [{paired + 1}/{max_volumes}] {stem}  "
            f"shape={img_data.shape}  "
            f"labels={list(nz_labels)}  "
            f"HU=[{img_data.min():.0f}, {img_data.max():.0f}]"
        )

        dataset[pid] = {
            "image": img_data,
            "segmentations": mask_data,
            "modality": MODALITY,
        }
        paired += 1

    print(
        f"\nPaired: {paired}  |  missing / errors: {missing}"
    )

    # -------------------------------------------------------------------
    # 4. Save
    # -------------------------------------------------------------------
    if dataset:
        print(f"\nSaving {len(dataset)} volumes to {output_path}.npz (uncompressed for mmap) …")
        _save_dataset_npz(dataset, output_path)
        print(f"Label map: {dict(LABEL_MAP)}")
    else:
        print("\nError: No volumes were successfully paired!")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Couinaud Liver NIfTI → .npz converter"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename stem without .npz extension (default: couinaud_liver)",
    )
    parser.add_argument(
        "--max-volumes",
        type=int,
        default=40,
        help="Maximum number of volumes to include (default: 40)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for HuggingFace downloads (default: _couinaud_cache next to script)",
    )

    args = parser.parse_args()

    out = args.output or "couinaud_liver"

    process_couinaud_dataset(
        output_path=out,
        max_volumes=args.max_volumes,
        cache_dir=args.cache_dir,
    )
