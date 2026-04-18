"""
flare_to_npz.py
================
Converts FLARE CT data (.nii.gz) into .npz format.
Anatomical cropping is provided by utils.cropping.
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
from utils.cropping import crop_to_anatomy, ct_signal_threshold
from utils.itk_utils import load_medical_image, get_voxel_sizes
import numpy as np
import SimpleITK as sitk



def process_flare_dataset(data_dir: str, output_path: str, crop: bool = True, margin: int = 15):
    """
    Process FLARE .nii.gz dataset into a .npz file.
    Optionally applies anatomical cropping.
    """
    dataset = {}
    modality = "CT"
    
    # Find all image paths (ends with _0000.nii.gz)
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*_0000.nii.gz")))
    
    for i, img_path in enumerate(image_paths):
        # Base names e.g., train_000_0000.nii.gz
        base_name = os.path.basename(img_path)
        p_idx = base_name.replace("_0000.nii.gz", "")
        mask_name = base_name.replace("_0000.nii.gz", ".nii.gz")
        mask_path = os.path.join(data_dir, mask_name)
        
        print(f"\n[{i + 1}/{len(image_paths)}] Processing {p_idx} ...")
        
        if not os.path.exists(mask_path):
            print(f"  Warning: Mask not found for {p_idx} at {mask_path}, skipping.")
            continue
            
        # 1. Load image and mask
        img_sitk, img_array = load_medical_image(img_path)
        
        mask_sitk, mask_array = load_medical_image(mask_path)
        
        # Verify shape
        if img_array.shape != mask_array.shape:
             print(f"  Warning: Shape mismatch for {p_idx}. Image: {img_array.shape}, Mask: {mask_array.shape}. Skipping.")
             continue
             
        # 2. Crop to anatomy (optional)
        if crop:
            print(f"  Cropping to anatomy (margin={margin}) ...")
            img_array, mask_array = crop_to_anatomy(
                img_array, mask_array, margin, threshold_fn=ct_signal_threshold
            )

        # 3. Resample to isotropic
        voxel_sizes = get_voxel_sizes(img_sitk)
        print(f"  Resampling from spacing {tuple(f'{v:.2f}' for v in voxel_sizes)} mm → 1.0 mm isotropic ...")
        
        img_resampled = resample_isotropic(img_array, voxel_sizes, is_mask=False, is_ct=True)
        mask_resampled = resample_isotropic(mask_array, voxel_sizes, is_mask=True, is_ct=False)
        mask_resampled = mask_resampled.astype(np.uint8)
        
        print(f"  -> Resampled shape: {img_resampled.shape}")
        
        # 4. Store in dataset dictionary
        dataset[p_idx] = {
            "image": img_resampled,
            "segmentations": mask_resampled,
            "modality": modality
        }
        
    if dataset:
        _save_dataset_npz(dataset, output_path)
    else:
        print("\nError: no patients were processed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FLARE .nii.gz → .npz converter")
    parser.add_argument("--data-dir", default=None, help="Path to FLARE data directory")
    parser.add_argument("--output", default="FLARE", help="Name or path for the output .npz file (without extension)")
    parser.add_argument("--no-crop", action="store_false", dest="crop", help="Disable anatomical cropping")
    parser.add_argument("--margin", type=int, default=15, help="Margin for cropping (default: 15)")
    parser.set_defaults(crop=True)
    args = parser.parse_args()
    
    # Make default data-dir resilient to where script is executed from
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir_path = os.path.join(script_dir, "FLARE")
    else:
        data_dir_path = args.data_dir
        
    if not os.path.isdir(data_dir_path):
        print(f"Error: Data directory not found: {data_dir_path}")
        sys.exit(1)
        
    process_flare_dataset(data_dir_path, args.output, crop=args.crop, margin=args.margin)
