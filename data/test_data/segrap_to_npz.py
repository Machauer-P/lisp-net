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

class SegRapProcessor:
    """
    Processes the SegRap2023 dataset into a unified .npz dataset.
    Supports processing either the non-contrast (ncCT) or contrast-enhanced (ceCT) scans.
    """
    def __init__(self, data_folder: str, modality: str = "ncct", margin: int = 15):
        self.data_folder = data_folder
        self.margin = margin
        self.modality = modality.lower()
        self.oar_mapping = {}
        self._next_oar_id = 1
        
    def _get_oar_id(self, oar_name: str) -> int:
        if oar_name not in self.oar_mapping:
            self.oar_mapping[oar_name] = self._next_oar_id
            self._next_oar_id += 1
        return self.oar_mapping[oar_name]

    def build_combined_mask(self, seg_files: list, ref_shape: tuple) -> np.ndarray:
        """
        Combine multiple individual organ masks into a single label map.
        To handle overlaps naturally (e.g. BrainStem overlapping Brain), 
        we apply masks from largest volume to smallest. We also ensure Gross Tumor Volumes (GTVs) 
        are applied last, giving them the highest priority.
        """
        combined = np.zeros(ref_shape, dtype=np.uint8)
        mask_data = []
        for seg_file in seg_files:
            oar_name = os.path.basename(seg_file).replace(".nii.gz", "")
            seg_img, seg_array = load_medical_image(seg_file)
            if seg_array.shape != ref_shape:
                print(f"      -> WARNING: shape mismatch for {oar_name} "
                      f"({seg_array.shape} vs {ref_shape}), skipping.")
                continue
                
            volume = np.sum(seg_array > 0)
            is_gtv = oar_name.startswith("GTV")
            mask_data.append({
                "name": oar_name,
                "array": seg_array,
                "volume": volume,
                "is_gtv": is_gtv
            })
            
        # Sort: false before true for GTV, then descending by volume.
        # This prioritizes smaller organs internally, and prioritizes GTV above OARs.
        mask_data.sort(key=lambda x: (x["is_gtv"], -x["volume"]))
        
        for md in mask_data:
            oar_id = self._get_oar_id(md["name"])
            write_mask = md["array"] > 0
            combined[write_mask] = oar_id
            
        return combined

    def process_dataset(self, output_filename: str):
        dataset = {}
        patient_folders = sorted([f.path for f in os.scandir(self.data_folder) if f.is_dir() and "segrap_" in f.name])
        
        if self.modality == "cect":
            img_file_name = "image_contrast.nii.gz"
            out_modality = "ceCT"
        else:
            img_file_name = "image.nii.gz"
            out_modality = "ncCT"

        for i, p_folder in enumerate(patient_folders):
            p_idx = os.path.basename(p_folder)
            print(f"\n[{i + 1}/{len(patient_folders)}] Processing {p_idx} ({out_modality}) ...")
            
            img_path = os.path.join(p_folder, img_file_name)
            if not os.path.exists(img_path):
                print(f"  Image not found for {p_idx} at {img_path}, skipping.")
                continue
                
            img_sitk, img_array = load_medical_image(img_path)
            
            # Find all segmentation files, excluding the original images
            all_files = sorted(glob.glob(os.path.join(p_folder, "*.nii.gz")))
            seg_files = [f for f in all_files if os.path.basename(f) not in ["image.nii.gz", "image_contrast.nii.gz"]]
            
            if not seg_files:
                print(f"  No valid segmentations for {p_idx}, skipping.")
                continue
                
            combined_mask = self.build_combined_mask(seg_files, img_array.shape)
            
            print(f"  Cropping to anatomy (margin={self.margin}) ...")
            img_array, combined_mask = crop_to_anatomy(
                img_array, combined_mask, self.margin, threshold_fn=ct_signal_threshold
            )
            
            voxel_sizes = get_voxel_sizes(img_sitk)
            print(f"  Resampling from spacing {tuple(f'{v:.2f}' for v in voxel_sizes)} mm → 1.0 mm isotropic ...")
            
            img_array = resample_isotropic(img_array, voxel_sizes, is_mask=False, is_ct=True)
            combined_mask = resample_isotropic(combined_mask, voxel_sizes, is_mask=True, is_ct=False)
            combined_mask = combined_mask.astype(np.uint8)
            
            print(f"  -> Resampled shape: {img_array.shape}")
            
            dataset[p_idx] = {
                "image": img_array,
                "segmentations": combined_mask,
                "modality": out_modality,
            }
            
        if dataset:
            _save_dataset_npz(dataset, output_filename)
            print(f"\nOAR Mapping: {self.oar_mapping}")
        else:
            print("\nError: no patients were processed!")
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SegRap2023 .nii.gz → .npz converter")
    parser.add_argument("modality", choices=["ncct", "cect"], nargs="?", default="ncct", help="Which modality to process (default: ncct)")
    parser.add_argument("--data-dir", default=None, help="Path to SegRap2023 data directory")
    parser.add_argument("--output", default=None, help="Name or path for the output .npz file (without extension)")
    parser.add_argument("--margin", type=int, default=15, help="Margin for cropping (default: 15)")
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir_path = os.path.join(script_dir, "SegRap2023")
    else:
        data_dir_path = args.data_dir
        
    out = args.output or f"SegRap2023_{args.modality.upper()}"
    
    proc = SegRapProcessor(data_dir_path, modality=args.modality, margin=args.margin)
    proc.process_dataset(out)
