"""
brats_to_npz.py
================
Converts BraTS 2024 challenge data (.nii.gz) into .npz format.

Supports two sub-challenges:
  • BraTS-GLI  – Adult glioma (post-treatment), 4 MRI sequences per patient.
  • BraTS-MEN-RT – Meningioma radiotherapy planning, 1 MRI sequence per patient.

Usage:
    python brats_to_npz.py gli     --data-dir BraTS_2024/BraTS-GLI     --output BraTS_GLI_t1c
    python brats_to_npz.py gli     --data-dir BraTS_2024/BraTS-GLI     --output BraTS_GLI_t1c --sequence t1c
    python brats_to_npz.py men_rt  --data-dir BraTS_2024/BraTS-MEN-RT  --output BraTS_MEN_RT

----------------------------------------------------------------------
IMPORTANT NOTES FROM THE PAPERS
----------------------------------------------------------------------
BraTS-GLI (arXiv 2405.18368):
  • 4 co-registered, skull-stripped, 1 mm isotropic sequences per case:
      t1c  – post-contrast T1-weighted
      t1n  – native (pre-contrast) T1-weighted
      t2f  – T2 FLAIR (fluid-attenuated inversion recovery)
      t2w  – T2-weighted
  • All volumes already in 1 mm isotropic space, standardised to 240×240×155 voxels.
    → No resampling strictly needed, but we keep it in the pipeline to ensure
      consistency if future data diverges from the standard.
  • Segmentation labels (integer values in *-seg.nii.gz):
      0 – Background
      1 – NETC  Non-Enhancing Tumor Core   (replaces old NCR+NET)
      2 – SNFH  Surrounding Non-enhancing FLAIR Hyperintensity  (replaces ED)
      3 – ET    Enhancing Tissue
      4 – RC    Resection Cavity            (NEW in 2024 post-treatment challenge)
  • Not all four labels are present in every patient (e.g. label 4 absent in
    treatment-naïve cases / recurrence not yet resected).
  • One of the 4 sequences is selected per run (default: t1c). If `--sequence all` is
    requested, all 4 contrasts are extracted and stored as independent 1-channel images
    (e.g., patientID_t1c, patientID_t2f) within a single consolidated .npz file.

BraTS-MEN-RT (arXiv 2405.18383):
  • Only T1c (post-contrast) is available; images are in NATIVE acquisition space.
  • NOT skull-stripped; facial features are defaced for privacy.
  • Voxel spacing and volume dimensions vary between patients  ← critical for loader.
  • Segmentation label (integer values in *_gtv.nii.gz):
      0 – Background
      1 – Target Volume (Gross Tumor Volume + post-operative at-risk regions)
  • Because data is in native space, resampling to 1 mm isotropic is applied here
    (same as FLARE/HanSeg pipeline) to normalise across patients.
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
from utils.cropping import crop_to_anatomy, mri_signal_threshold
from utils.itk_utils import load_medical_image, get_voxel_sizes
import numpy as np
import SimpleITK as sitk


# ======================================================================
# Shared helpers
# ======================================================================

# Using shared load_medical_image and get_voxel_sizes from utils.itk_utils



# ======================================================================
# BraTS-GLI processor
# ======================================================================

BRATS_GLI_SEQUENCES = ("t1c", "t1n", "t2f", "t2w")

# Segmentation label map (from the challenge paper)
BRATS_GLI_LABEL_MAP = {
    0: "Background",
    1: "NETC",   # Non-Enhancing Tumor Core
    2: "SNFH",   # Surrounding Non-enhancing FLAIR Hyperintensity
    3: "ET",     # Enhancing Tissue
    4: "RC",     # Resection Cavity (new in BraTS 2024 post-treatment)
}


def process_brats_gli(
    data_dir: str,
    output_path: str,
    sequence: str = "t1c",
    crop: bool = True,
    margin: int = 10,
    resample: bool = True,
    max_patients: int = None,
):
    if sequence not in (*BRATS_GLI_SEQUENCES, "all"):
        raise ValueError(
            f"sequence must be one of {(*BRATS_GLI_SEQUENCES, 'all')}, got '{sequence}'"
        )

    dataset = {}
    patient_folders = sorted(
        [f.path for f in os.scandir(data_dir) if f.is_dir()]
    )
    if max_patients is not None and max_patients > 0 and len(patient_folders) > max_patients:
        import random
        # Setting seed to ensure reproducible subsets if rerun
        random.seed(42)
        patient_folders = random.sample(patient_folders, max_patients)

    for i, p_folder in enumerate(patient_folders):
        p_idx = os.path.basename(p_folder)
        print(f"\n[{i + 1}/{len(patient_folders)}] Processing {p_idx} …")

        seg_paths = glob.glob(os.path.join(p_folder, "*-seg.nii.gz"))
        if not seg_paths:
            print(f"  WARNING: no segmentation found for {p_idx}, skipping.")
            continue

        seg_sitk, seg_array = load_medical_image(seg_paths[0], dtype=np.uint8)
        present = sorted(int(v) for v in np.unique(seg_array) if v != 0)
        label_names = [BRATS_GLI_LABEL_MAP.get(v, f"label_{v}") for v in present]
        print(f"  Labels present: {dict(zip(present, label_names))}")

        if sequence == "all":
            channels = []
            ref_sitk = None
            for seq in BRATS_GLI_SEQUENCES:
                seq_paths = glob.glob(os.path.join(p_folder, f"*-{seq}.nii.gz"))
                if not seq_paths:
                    print(f"  WARNING: sequence {seq} missing for {p_idx}, skipping parameter.")
                    continue
                s_sitk, s_arr = load_medical_image(seq_paths[0])
                if ref_sitk is None:
                    ref_sitk = s_sitk
                channels.append((seq, s_arr))
            
            if not channels:
                continue
                
            for seq, img_arr in channels:
                if img_arr.shape[-3:] != seg_array.shape:
                    print(f"  WARNING: shape mismatch for {seq} – skipping.")
                    continue
                    
                cur_img = img_arr.copy()
                cur_seg = seg_array.copy()
                
                if crop:
                    cur_img, cur_seg = crop_to_anatomy(
                        cur_img, cur_seg, margin, threshold_fn=mri_signal_threshold
                    )
                    
                if resample:
                    vox = get_voxel_sizes(ref_sitk)
                    cur_img = resample_isotropic(cur_img, vox, is_mask=False, is_ct=False)
                    cur_seg = resample_isotropic(cur_seg, vox, is_mask=True, is_ct=False)
                    cur_seg = cur_seg.astype(np.uint8)
                    
                dataset[f"{p_idx}_{seq}"] = {
                    "image":         cur_img,
                    "segmentations": cur_seg,
                    "modality":      "MRI",
                }
            print(f"  → Success: {p_idx} (saved {len(channels)} contrasts individually)")

        else:
            seq_paths = glob.glob(os.path.join(p_folder, f"*-{sequence}.nii.gz"))
            if not seq_paths:
                print(f"  WARNING: sequence '{sequence}' missing for {p_idx}, skipping.")
                continue
            ref_sitk, image_array = load_medical_image(seq_paths[0])

            if image_array.shape[-3:] != seg_array.shape:
                print(f"  WARNING: shape mismatch – skipping.")
                continue

            if crop:
                print(f"  Cropping to brain bounding box (margin={margin}) …")
                image_array, seg_array = crop_to_anatomy(
                    image_array, seg_array, margin, threshold_fn=mri_signal_threshold
                )

            if resample:
                vox = get_voxel_sizes(ref_sitk)
                print(f"  Spacing: {tuple(f'{v:.3f}' for v in vox)} mm (target: 1.0 mm isotropic)")
                image_array = resample_isotropic(image_array, vox, is_mask=False, is_ct=False)
                seg_array   = resample_isotropic(seg_array,   vox, is_mask=True,  is_ct=False)
                seg_array   = seg_array.astype(np.uint8)
                print(f"  → Resampled shape: {image_array.shape}")
            else:
                print(f"  Shape: {image_array.shape}  (resampling skipped)")

            dataset[p_idx] = {
                "image":         image_array,
                "segmentations": seg_array,
                "modality":      "MRI",
            }
            print(f"  → Success: {p_idx}  shape={image_array.shape}")

    if dataset:
        _save_dataset_npz(dataset, output_path)
        print(f"\nLabel map: {BRATS_GLI_LABEL_MAP}")
    else:
        print("\nError: no patients were processed!")

# ======================================================================
# BraTS-MEN-RT processor
# ======================================================================

# Segmentation label map (from the challenge paper)
BRATS_MEN_RT_LABEL_MAP = {
    0: "Background",
    1: "Target Volume",  # GTV + post-operative at-risk sites
}


def process_brats_men_rt(
    data_dir: str,
    output_path: str,
    crop: bool = True,
    margin: int = 15,
    resample: bool = True,
    max_patients: int = None,
):
    """
    Process BraTS-MEN-RT (.nii.gz) dataset into a .npz file.

    Parameters
    ----------
    data_dir   : Root directory containing one sub-folder per patient.
    output_path: Output filename (without .npz).
    crop       : Apply anatomical cropping.
                 NOTE: MEN-RT images are NOT skull-stripped (brain, skull, face/
                 defaced region included).  The signal threshold will still find
                 the bright contrast-enhancing tumour region, but may include
                 skull/scalp.  Crop primarily saves memory.
    margin     : Voxel margin for cropping.
    resample   : Resample to 1 mm isotropic.
                 MEN-RT images are in NATIVE acquisition space (spacing varies
                 across patients, often ~1×1×1 mm but not guaranteed).
                 Enable this to normalise across patients.
    """
    dataset = {}
    patient_folders = sorted(
        [f.path for f in os.scandir(data_dir) if f.is_dir()]
    )
    if max_patients is not None and max_patients > 0 and len(patient_folders) > max_patients:
        import random
        random.seed(42)
        patient_folders = random.sample(patient_folders, max_patients)

    for i, p_folder in enumerate(patient_folders):
        p_idx = os.path.basename(p_folder)
        print(f"\n[{i + 1}/{len(patient_folders)}] Processing {p_idx} …")

        # ------------------------------------------------------------------
        # 1. Locate T1c image  (*_t1c.nii.gz)
        # ------------------------------------------------------------------
        t1c_paths = glob.glob(os.path.join(p_folder, "*_t1c.nii.gz"))
        if not t1c_paths:
            print(f"  WARNING: T1c image not found for {p_idx}, skipping.")
            continue

        img_sitk, img_array = load_medical_image(t1c_paths[0])

        # ------------------------------------------------------------------
        # 2. Locate GTV segmentation  (*_gtv.nii.gz)
        # ------------------------------------------------------------------
        gtv_paths = glob.glob(os.path.join(p_folder, "*_gtv.nii.gz"))
        if not gtv_paths:
            print(f"  WARNING: GTV segmentation not found for {p_idx}, skipping.")
            continue

        _, seg_array = load_medical_image(gtv_paths[0], dtype=np.uint8)

        # Verify matching shapes
        if img_array.shape != seg_array.shape:
            print(
                f"  WARNING: shape mismatch  image={img_array.shape}  "
                f"seg={seg_array.shape}  – skipping."
            )
            continue

        # Report GTV presence (occasionally empty in post-op cases)
        gtv_vox = int(np.sum(seg_array > 0))
        print(f"  GTV voxels: {gtv_vox}")
        if gtv_vox == 0:
            print(f"  WARNING: empty GTV mask for {p_idx}.  Patient kept but mask is all-zero.")

        # ------------------------------------------------------------------
        # 3. Log native voxel spacing (critical – MEN-RT is in native space!)
        # ------------------------------------------------------------------
        vox = get_voxel_sizes(img_sitk)
        print(f"  Native spacing: {tuple(f'{v:.3f}' for v in vox)} mm  |  shape: {img_array.shape}")

        # ------------------------------------------------------------------
        # 4. Anatomical cropping
        #    Uses the T1c signal to find tissue boundaries.
        #    NOTE: MEN-RT is NOT skull-stripped, so the crop bounding box may
        #    include skull and scalp tissue.  This is expected and correct –
        #    skull context is needed for RT planning target localisation.
        # ------------------------------------------------------------------
        # if crop:
        #     print(f"  Cropping to anatomy (margin={margin}) …")
        #     img_array, seg_array = crop_to_anatomy(
        #         img_array, seg_array, margin, threshold_fn=mri_signal_threshold
        #     )
        #     print(f"  → Cropped shape: {img_array.shape}")

        # ------------------------------------------------------------------
        # 5. Resample to 1 mm isotropic
        #    Critical for MEN-RT: native spacing varies across institutions/scanners.
        # ------------------------------------------------------------------
        if resample:
            print(
                f"  Resampling from {tuple(f'{v:.3f}' for v in vox)} mm → 1.0 mm isotropic …"
            )
            img_array = resample_isotropic(img_array, vox, is_mask=False, is_ct=False)
            seg_array = resample_isotropic(seg_array, vox, is_mask=True,  is_ct=False)
            seg_array = seg_array.astype(np.uint8)
            print(f"  → Resampled shape: {img_array.shape}")
        else:
            print(f"  Shape: {img_array.shape}  (resampling skipped – WARNING: spacing varies!)")

        # ------------------------------------------------------------------
        # 6. Store
        # ------------------------------------------------------------------
        dataset[p_idx] = {
            "image":         img_array,
            "segmentations": seg_array,
            "modality":      "MRI",
        }
        print(f"  → Success: {p_idx}  shape={img_array.shape}")

    if dataset:
        _save_dataset_npz(dataset, output_path)
        print(f"\nLabel map: {BRATS_MEN_RT_LABEL_MAP}")
    else:
        print("\nError: no patients were processed!")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BraTS 2024 → .npz converter (GLI or MEN-RT)"
    )
    subparsers = parser.add_subparsers(dest="challenge", required=True)

    # ── BraTS-GLI ──────────────────────────────────────────────────────
    gli_p = subparsers.add_parser(
        "gli",
        help="Process BraTS-GLI adult glioma dataset",
    )
    gli_p.add_argument(
        "--data-dir",
        default=None,
        help="Path to BraTS-GLI directory (one sub-folder per patient)",
    )
    gli_p.add_argument(
        "--output",
        default="BraTS_GLI",
        help="Output filename stem (without .npz)",
    )
    gli_p.add_argument(
        "--sequence",
        default="t1c",
        choices=[*BRATS_GLI_SEQUENCES, "all"],
        help=(
            "MRI sequence to use as image.  "
            "'all' saves all 4 contrasts independently "
            "as 1-channel arrays in the .npz output."
        ),
    )
    gli_p.add_argument(
        "--no-resample",
        action="store_false",
        dest="resample",
        help="Skip resampling (GLI is already 1 mm isotropic)",
    )
    gli_p.add_argument(
        "--crop",
        action="store_true",
        default=False,
        help="Enable anatomical cropping (disabled by default – see commented code)",
    )
    gli_p.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Voxel margin for cropping (default: 10)",
    )
    gli_p.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of random patients to process. If set, randomly samples this many folders.",
    )
    gli_p.set_defaults(resample=True)

    # ── BraTS-MEN-RT ───────────────────────────────────────────────────
    men_p = subparsers.add_parser(
        "men_rt",
        help="Process BraTS-MEN-RT meningioma radiotherapy dataset",
    )
    men_p.add_argument(
        "--data-dir",
        default=None,
        help="Path to BraTS-MEN-RT directory (one sub-folder per patient)",
    )
    men_p.add_argument(
        "--output",
        default="BraTS_MEN_RT",
        help="Output filename stem (without .npz)",
    )
    men_p.add_argument(
        "--no-resample",
        action="store_false",
        dest="resample",
        help=(
            "Skip resampling.  WARNING: MEN-RT is in native space "
            "(variable spacing) – only skip if you handle this downstream."
        ),
    )
    men_p.add_argument(
        "--crop",
        action="store_true",
        default=False,
        help="Enable anatomical cropping (disabled by default – see commented code)",
    )
    men_p.add_argument(
        "--margin",
        type=int,
        default=15,
        help="Voxel margin for cropping (default: 15)",
    )
    men_p.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of random patients to process. If set, randomly samples this many folders.",
    )
    men_p.set_defaults(resample=True)

    args = parser.parse_args()

    # Resolve default data directories relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.challenge == "gli":
        if args.data_dir is None:
            args.data_dir = os.path.join(script_dir, "BraTS_2024", "BraTS-GLI")
        if not os.path.isdir(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}")
            sys.exit(1)
        process_brats_gli(
            data_dir=args.data_dir,
            output_path=args.output,
            sequence=args.sequence,
            crop=args.crop,
            margin=args.margin,
            resample=args.resample,
            max_patients=args.max_patients,
        )

    elif args.challenge == "men_rt":
        if args.data_dir is None:
            args.data_dir = os.path.join(script_dir, "BraTS_2024", "BraTS-MEN-RT")
        if not os.path.isdir(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}")
            sys.exit(1)
        process_brats_men_rt(
            data_dir=args.data_dir,
            output_path=args.output,
            crop=args.crop,
            margin=args.margin,
            resample=args.resample,
            max_patients=args.max_patients,
        )
