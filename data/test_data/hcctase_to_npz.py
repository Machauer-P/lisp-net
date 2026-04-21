"""
HCC-TACE-Seg dataset → .npz converter
======================================
Dataset: Pre-procedural multiphasic contrast-enhanced CT with liver, tumor,
         portal vein and aorta segmentations for 105 HCC patients.

Source format:
  hcc_tace_seg/
    HCC_XXX/
      <StudyUID>/
        <SeriesUID>/       # CT series (PRE, arterial/portal 3-phase)
          *.dcm
        <SEG_SeriesUID>/   # DICOM-SEG (1 file)
          *.dcm

Each patient has:
  - 1 DICOM-SEG file containing 4 segments: Liver, Mass, Portal vein, Abdominal aorta
  - Several CT series:
      "PRE LIVER"        : non-contrast baseline (~20-27 slices)
      "LIVER 3 PHASE"    : multiphasic contrast (arterial+portal stacked, ~100-200+ slices)

Strategy:
  The SEG was annotated on the 3-phase contrast series. We find the CT series
  that the SEG references (via ReferencedSeriesSequence) and pair them.
  If the SEG covers only half the slices (one phase), we use the matching half.

  OAR IDs:  1=Liver, 2=Mass, 3=Portal vein, 4=Abdominal aorta
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import SimpleITK as sitk
import pydicom
import highdicom as hd

from data.test_data.ds_handler import save_dataset as _save_dataset_npz
from utils.resampling import resample_isotropic
from utils.cropping import crop_to_anatomy, ct_signal_threshold
from utils.itk_utils import get_voxel_sizes

# Fixed OAR mapping matching segment order in SEG file
OAR_MAPPING = {
    "Liver": 1,
    "Mass": 2,
    "Portal vein": 3,
    "Abdominal aorta": 4,
}
# Priority for overlapping regions (Lowest first, highest last)
# Liver is the background, vessels run through it, mass is the primary target.
PRIORITY_ORDER = ["Liver", "Abdominal aorta", "Portal vein", "Mass"]


def find_seg_file(patient_folder: str) -> str | None:
    """Find the single DICOM-SEG file in a patient folder."""
    for dcm_file in glob.glob(os.path.join(patient_folder, "**", "*.dcm"), recursive=True):
        ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
        if getattr(ds, "Modality", "") == "SEG":
            return dcm_file
    return None


def get_ordered_ct_files_from_seg(patient_folder: str, seg: hd.seg.Segmentation) -> list[tuple]:
    """Find the exact CT dicom files referenced by the SEG, sorted by Z coordinate."""
    # 1. Extract all referenced SOPInstanceUIDs from the SEG
    ref_sops = set()
    for frame in seg.PerFrameFunctionalGroupsSequence:
        ref_sops.add(frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
        
    # 2. Find exactly the CT files that match those SOPInstanceUIDs
    ct_files = []
    for dcm_file in glob.glob(os.path.join(patient_folder, "**", "*.dcm"), recursive=True):
        ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
        if getattr(ds, "Modality", "") == "CT" and ds.SOPInstanceUID in ref_sops:
            ct_files.append((dcm_file, ds.SOPInstanceUID, float(ds.ImagePositionPatient[2])))
            
    # 3. Sort by Z-coordinate ascending (standard ITK/DICOM spatial ordering)
    ct_files.sort(key=lambda x: x[2])
    return ct_files


def build_paired_volumes(seg: hd.seg.Segmentation, ct_files: list) -> tuple[sitk.Image, np.ndarray, np.ndarray]:
    """
    Constructs perfectly aligned CT and Label Map volumes by mapping DICOM-SEG 
    frames directly onto the Z-index of the ordered CT slices.
    """
    # Load CT volume
    ordered_paths = [x[0] for x in ct_files]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ordered_paths)
    ct_img = reader.Execute()
    ct_arr = sitk.GetArrayFromImage(ct_img)
    
    # Map SOPInstanceUID to Z-index in the array
    sop_to_z = {x[1]: i for i, x in enumerate(ct_files)}
    
    # Build raw multi-channel SEG array (Z, H, W, n_segments)
    label_names = [s.SegmentLabel for s in seg.SegmentSequence]
    raw_seg = np.zeros(ct_arr.shape + (len(label_names),), dtype=np.uint8)
    
    # Loop each frame and map it exactly onto its true Z slice mathematically
    for i, frame in enumerate(seg.PerFrameFunctionalGroupsSequence):
        sop = frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        segment_number = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        z = sop_to_z.get(sop)
        if z is not None:
            layer = segment_number - 1  # 0-indexed
            raw_seg[z, :, :, layer] = seg.pixel_array[i]
        
    # Collapse into 1D label map using priority order
    combined = np.zeros(ct_arr.shape, dtype=np.uint8)
    for name in PRIORITY_ORDER:
        if name in label_names:
            idx = label_names.index(name)
            oar_id = OAR_MAPPING.get(name, len(OAR_MAPPING) + 1)
            combined[raw_seg[..., idx] == 1] = oar_id
            
    return ct_img, ct_arr, combined


def process_hcctase_dataset(data_folder: str, output_filename: str, margin: int = 15):
    patient_folders = sorted(
        [f.path for f in os.scandir(data_folder) if f.is_dir() and "HCC_" in f.name]
    )

    dataset = {}
    skipped = []

    for i, p_folder in enumerate(patient_folders):
        p_id = os.path.basename(p_folder)
        print(f"\n[{i + 1}/{len(patient_folders)}] Processing {p_id} ...")

        # 1. Find the SEG file
        seg_path = find_seg_file(p_folder)
        if seg_path is None:
            print(f"  WARNING: No SEG file found for {p_id}, skipping.")
            skipped.append(p_id)
            continue

        # 2. Read SEG
        seg = hd.seg.segread(seg_path)
        label_names = [d.SegmentLabel for d in seg.SegmentSequence]
        print(f"  SEG segments: {label_names}")

        if not hasattr(seg, "ReferencedSeriesSequence") or not seg.ReferencedSeriesSequence:
            print(f"  WARNING: SEG has no ReferencedSeriesSequence for {p_id}, skipping.")
            skipped.append(p_id)
            continue

        ref_series_uid = seg.ReferencedSeriesSequence[0].SeriesInstanceUID
        print(f"  SEG references CT series: ...{ref_series_uid[-32:]}")

        # 3 + 4 + 5 + 6. Construct mathematically aligned and paired volumes
        try:
            ct_files = get_ordered_ct_files_from_seg(p_folder, seg)
            if not ct_files:
                print(f"  WARNING: Could not find referenced CT files for {p_id}, skipping.")
                skipped.append(p_id)
                continue
                
            ct_img, ct_arr, label_map = build_paired_volumes(seg, ct_files)
        except Exception as e:
            print(f"  WARNING: Failed to align CT/SEG for {p_id}: {e}, skipping.")
            skipped.append(p_id)
            continue

        # 7. Crop to anatomy
        print(f"  Cropping to anatomy (margin={margin}) ...")
        try:
            ct_arr, label_map = crop_to_anatomy(
                ct_arr, label_map, margin, threshold_fn=ct_signal_threshold
            )
        except Exception as e:
            print(f"  WARNING: Cropping failed ({e}), using uncropped volume.")

        # 8. Resample to isotropic 1.0 mm
        voxel_sizes = get_voxel_sizes(ct_img)
        print(f"  Resampling from {tuple(f'{v:.3f}' for v in voxel_sizes)} -> 1.0 mm isotropic ...")
        ct_arr = resample_isotropic(ct_arr, voxel_sizes, is_mask=False, is_ct=True)
        label_map = resample_isotropic(label_map, voxel_sizes, is_mask=True, is_ct=False)
        label_map = label_map.astype(np.uint8)
        print(f"  -> Final shape: {ct_arr.shape}")

        dataset[p_id] = {
            "image": ct_arr,
            "segmentations": label_map,
            "modality": "ceCT",
        }

    if dataset:
        _save_dataset_npz(dataset, output_filename)
        print(f"\nOAR mapping used: {OAR_MAPPING}")
        if skipped:
            print(f"Skipped patients: {skipped}")
    else:
        print("\nError: no patients were successfully processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCC-TACE-Seg DICOM → .npz converter")
    parser.add_argument("--data-dir", default=None, help="Path to hcc_tace_seg directory")
    parser.add_argument("--output", default=None, help="Output .npz filename (without extension)")
    parser.add_argument("--margin", type=int, default=15, help="Crop margin in voxels (default: 15)")
    args = parser.parse_args()

    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir_path = os.path.join(script_dir, "HCCTase", "hcc_tace_seg")
    else:
        data_dir_path = args.data_dir

    out = args.output or "HCCTase_ceCT"

    process_hcctase_dataset(data_dir_path, out, margin=args.margin)
