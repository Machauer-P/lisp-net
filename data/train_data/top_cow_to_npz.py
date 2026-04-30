"""
top_cow_to_npz.py
=================
Converts TopCoW 2024 challenge data (.nii.gz) into .npz format.

Supports two modalities:
  • CT
  • MR

Usage:
    python top_cow_to_npz.py ct --data-dir TopCoW_2024 --output TopCoW_CT
    python top_cow_to_npz.py mr --data-dir TopCoW_2024 --output TopCoW_MR
"""

import os
import glob
import sys
from pathlib import Path

# Ensure project root is on sys.path so utils can be imported
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.test_data.ds_handler import save_dataset as _save_dataset_npz
from utils.resampling import resample_isotropic
from utils.cropping import crop_to_anatomy, mri_signal_threshold, ct_signal_threshold
from utils.itk_utils import load_medical_image, get_voxel_sizes
import numpy as np

TOPCOW_LABEL_MAP = {
    0: "Background",
    1: "BA",
    2: "R-PCA",
    3: "L-PCA",
    4: "R-ICA",
    5: "R-MCA",
    6: "L-ICA",
    7: "L-MCA",
    8: "R-Pcom",
    9: "L-Pcom",
    10: "Acom",
    11: "R-ACA",
    12: "L-ACA",
    15: "3rd-A2"
}

def process_topcow(
    data_dir: str,
    output_path: str,
    modality: str = "ct",
    crop: bool = True,
    margin: int = 15,
    resample: bool = True,
    max_patients: int = None,
):
    dataset = {}
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "cow_seg_labelsTr")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"Error: Missing imagesTr or cow_seg_labelsTr in {data_dir}")
        return

    # Find all images for the given modality
    search_pattern = os.path.join(images_dir, f"topcow_{modality}_*_0000.nii.gz")
    image_paths = sorted(glob.glob(search_pattern))

    if max_patients is not None and max_patients > 0 and len(image_paths) > max_patients:
        import random
        image_paths = random.sample(image_paths, max_patients)
        image_paths.sort()

    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        # Extract patient ID, e.g., from topcow_ct_001_0000.nii.gz -> 001
        p_idx = filename.replace(f"topcow_{modality}_", "").replace("_0000.nii.gz", "")
        patient_key = f"topcow_{modality}_{p_idx}"
        
        print(f"\n[{i + 1}/{len(image_paths)}] Processing {patient_key} ...")

        label_path = os.path.join(labels_dir, f"topcow_{modality}_{p_idx}.nii.gz")
        if not os.path.exists(label_path):
            print(f"  WARNING: Label not found at {label_path}, skipping.")
            continue

        img_sitk, img_array = load_medical_image(img_path)
        _, seg_array = load_medical_image(label_path, dtype=np.uint8)

        if img_array.shape != seg_array.shape:
            print(f"  WARNING: Shape mismatch image={img_array.shape} seg={seg_array.shape} - skipping.")
            continue

        vox = get_voxel_sizes(img_sitk)
        print(f"  Native spacing: {tuple(f'{v:.3f}' for v in vox)} mm | shape: {img_array.shape}")

        if crop:
            print(f"  Cropping to anatomy (margin={margin}) ...")
            threshold_fn = ct_signal_threshold if modality == "ct" else mri_signal_threshold
            img_array, seg_array = crop_to_anatomy(
                img_array, seg_array, margin, threshold_fn=threshold_fn
            )
            print(f"  -> Cropped shape: {img_array.shape}")

        if resample:
            print(f"  Resampling from {tuple(f'{v:.3f}' for v in vox)} mm -> 1.0 mm isotropic ...")
            is_ct = (modality == "ct")
            img_array = resample_isotropic(img_array, vox, is_mask=False, is_ct=is_ct)
            seg_array = resample_isotropic(seg_array, vox, is_mask=True, is_ct=False)
            seg_array = seg_array.astype(np.uint8)
            print(f"  -> Resampled shape: {img_array.shape}")

        modality_tag = "MRI" if modality.lower() == "mr" else "CT"

        dataset[patient_key] = {
            "image": img_array,
            "segmentations": seg_array,
            "modality": modality_tag,
        }
        print(f"  -> Success: {patient_key}")

    if dataset:
        _save_dataset_npz(dataset, output_path)
        print(f"\nLabel map: {TOPCOW_LABEL_MAP}")
    else:
        print("\nError: no patients were processed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TopCoW 2024 -> .npz converter")
    parser.add_argument("modality", choices=["ct", "mr"], help="Modality to process (ct or mr)")
    parser.add_argument("--data-dir", default=None, help="Path to TopCoW_2024 directory")
    parser.add_argument("--output", default=None, help="Output filename stem (without .npz)")
    parser.add_argument("--no-resample", action="store_false", dest="resample", help="Skip resampling")
    parser.add_argument("--no-crop", action="store_false", dest="crop", help="Skip cropping")
    parser.add_argument("--margin", type=int, default=15, help="Voxel margin for cropping")
    parser.add_argument("--max-patients", type=int, default=None, help="Max patients to process")
    
    parser.set_defaults(resample=True, crop=True)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir is None:
        args.data_dir = os.path.join(script_dir, "TopCoW_2024")
    
    if args.output is None:
        args.output = f"TopCoW_{args.modality.upper()}"

    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    process_topcow(
        data_dir=args.data_dir,
        output_path=args.output,
        modality=args.modality,
        crop=args.crop,
        margin=args.margin,
        resample=args.resample,
        max_patients=args.max_patients,
    )
