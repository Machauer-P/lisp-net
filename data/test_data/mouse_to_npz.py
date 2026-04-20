import os
import sys
import glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# Ensure project root is on sys.path so utils can be imported
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.test_data.ds_handler import save_dataset as _save_dataset_npz
from utils.itk_utils import load_medical_image

MOUSE_ORGAN1_LABEL_MAP = {
    0: "unclassified",
    1: "Bone",
    2: "Lung",
    3: "Heart",
    4: "Liver",
    5: "Intestine",
    6: "Bladder",
    7: "Spleen",
    8: "Stomach",
    9: "Muscle",
    10: "Kidneys"
}

MOUSE_ORGAN2_LABEL_MAP = {
    0: "unclassified",
    1: "Bone",
    2: "Lung",
    3: "Trachea",
    4: "Heart",
    5: "Bladder",
    6: "Kidneys",
    7: "Stomach",
    8: "Muscle",
    9: "Spleen",
    10: "Liver",
    11: "Intestine"
}

def process_mouse_longitudinal(
    data_dir: str,
    output_path: str,
    organ_version: int = 2,
):
    """
    Process Mouse Longitudinal CT (.hdr / .img) dataset into a .npz file.
    Iterates over all 140 time-point subfolders and compiles them.

    Parameters
    ----------
    data_dir  : Root directory containing 'M01_0.25h', etc.
    output_path: Output filename (without .npz).
    organ_version: Which mask version to use (1 or 2).
    """

    dataset = {}
    
    # Locate all patient time-point subdirectories
    subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
    
    if not subfolders:
        print(f"Error: No subfolders found in {data_dir}.")
        sys.exit(1)

    print(f"Discovered {len(subfolders)} longitudinal scan folders. Processing...")
    
    organ_prefix = f"Organ{organ_version}"
    label_map = MOUSE_ORGAN2_LABEL_MAP if organ_version == 2 else MOUSE_ORGAN1_LABEL_MAP
    
    # Track statistics
    success_count = 0
    missing_count = 0

    for i, p_folder in enumerate(subfolders):
        p_idx = os.path.basename(p_folder)
        
        ct_hdr_list = glob.glob(os.path.join(p_folder, "CT280*.hdr"))
        organ_hdr_list = glob.glob(os.path.join(p_folder, f"{organ_prefix}*.hdr"))

        if not ct_hdr_list or not organ_hdr_list:
            print(f"  [{i+1}/{len(subfolders)}] {p_idx} - WARNING: Missing CT280 or {organ_prefix}. Skipping.")
            missing_count += 1
            continue
            
        ct_hdr = ct_hdr_list[0]
        organ_hdr = organ_hdr_list[0]

        print(f"  [{i+1}/{len(subfolders)}] Processing {p_idx} ...")

        # Load Atlas Masks
        seg_sitk, seg_array = load_medical_image(organ_hdr, dtype=np.uint8)

        # Load CT Image
        img_sitk, img_array = load_medical_image(ct_hdr, dtype=np.float32)

        # Ensure isotropic arrays are matched
        if img_array.shape != seg_array.shape:
            print(f"    WARNING: Shape mismatch image={img_array.shape} seg={seg_array.shape}. Skipping.")
            continue
            
        dataset[p_idx] = {
            "image":         img_array,
            "segmentations": seg_array,
            "modality":      "CT",
        }
        success_count += 1

    if dataset:
        print(f"\\nSaving {success_count} aggregated scans to {output_path}.npz ...")
        _save_dataset_npz(dataset, output_path)
        print("Save complete!")
        print(f"Label map used ({organ_prefix}): {label_map}")
    else:
        print("\\nError: No scans were processed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mouse Longitudinal CT Dataset → .npz converter"
    )
    
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to Mouse directory containing longitudinal subfolders",
    )
    parser.add_argument(
        "--output",
        default="Mouse_Longitudinal",
        help="Output filename stem (without .npz)",
    )
    parser.add_argument(
        "--organ",
        type=int,
        default=2,
        choices=[1, 2],
        help="Which mask version to load: Organ1 (10 classes) or Organ2 (11 classes incl. Trachea). Defaults to 2.",
    )

    args = parser.parse_args()

    # Resolve default data directories relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.data_dir is None:
        args.data_dir = os.path.join(script_dir, "Mouse")
        
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
        
    process_mouse_longitudinal(
        data_dir=args.data_dir,
        output_path=args.output,
        organ_version=args.organ,
    )
