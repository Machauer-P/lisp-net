"""
HanSeg_to_npz.py
================
Unified CT / MRI → .npz pipeline using class inheritance.

Usage:
    python HanSeg_to_npz.py ct  --data-dir HanSeg/set_1 --output HanSeg_CT
    python HanSeg_to_npz.py mri --data-dir HanSeg/set_1 --output HanSeg_MRI
"""

import os
import glob
import sys
from pathlib import Path

# Ensure project root is on sys.path so utils.ds_handler can be imported
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.test_data.ds_handler import save_dataset as _save_dataset_npz
import numpy as np
import pandas as pd
import SimpleITK as sitk


# ======================================================================
# Base Processor  (shared by CT and MRI)
# ======================================================================

class BaseProcessor:
    """
    Shared preprocessing logic for HanSeg CT and MRI pipelines.

    Subclass must implement
    ----------------------
    load_image(patient_folder, patient_idx)
        -> (reference_sitk_image, image_numpy_array, need_resample_masks: bool)
           or None to skip the patient.

    Optionally override
    -------------------
    get_signal_threshold(image_array) -> float
        Used by crop_to_anatomy to separate tissue from background.
    """

    # Keywords that identify non-organ (hardware / structural) masks
    HARDWARE_BLOCKLIST = [
        "couch", "bed", "table", "external",
        "mask", "immobilization", "support", "artifact",
    ]

    # Maximum fraction of total volume an organ mask may occupy (5 %)
    MAX_ORGAN_VOLUME_FRACTION = 0.05

    # Maximum number of unique values for a valid binary mask
    MAX_UNIQUE_VALUES = 10

    def __init__(self, data_folder: str,
                 oar_csv_path: str = None,
                 margin: int = 15):
        self.data_folder = data_folder
        self.margin = margin

        # Global OAR mapping: organ name → integer class ID
        # (consistent across all patients in a single run)
        self.oar_mapping: dict = {}
        self._next_oar_id = 1

        # Optional OAR availability CSV
        self.oar_availability = None
        if oar_csv_path and os.path.exists(oar_csv_path):
            print(f"Loading OAR availability from {oar_csv_path}")
            df = pd.read_csv(oar_csv_path, sep=None, engine="python")
            df = df.set_index(df.columns[0])
            if df.index.dtype == "object":
                df.index = (
                    df.index.astype(str)
                    .str.extract(r"(\d+)")[0]
                    .astype(int)
                )
            self.oar_availability = df

    # ------------------------------------------------------------------
    # OAR ID management (consistent across all patients)
    # ------------------------------------------------------------------

    def _get_oar_id(self, oar_name: str) -> int:
        if oar_name not in self.oar_mapping:
            self.oar_mapping[oar_name] = self._next_oar_id
            self._next_oar_id += 1
        return self.oar_mapping[oar_name]

    # ------------------------------------------------------------------
    # Segmentation file discovery & filtering
    # ------------------------------------------------------------------

    def get_valid_segmentation_files(
        self, patient_idx: str, folder: str
    ) -> list:
        """
        Return only organic OAR mask files, filtering out:
          1. Hardware keywords (bed, couch, …)
          2. OARs marked unavailable in the CSV (if loaded)
        """
        all_seg_files = sorted(
            glob.glob(os.path.join(folder, f"{patient_idx}_OAR_*.seg.nrrd"))
        )

        valid = []
        for seg_file in all_seg_files:
            oar_name = (
                os.path.basename(seg_file)
                .split("_OAR_")[1]
                .replace(".seg.nrrd", "")
            )
            norm = oar_name.lower().replace("_", "").replace(" ", "")

            # 1. Blocklist check
            if any(word in norm for word in self.HARDWARE_BLOCKLIST):
                continue

            # 2. CSV availability check (if CSV was loaded)
            if self.oar_availability is not None:
                try:
                    case_num = int(patient_idx.split("_")[1])
                except (IndexError, ValueError):
                    case_num = None

                if case_num is not None and case_num in self.oar_availability.index:
                    patient_oars = self.oar_availability.loc[case_num]
                    matched_col = None
                    for col in patient_oars.index:
                        norm_col = (
                            str(col).lower().replace("_", "").replace(" ", "")
                        )
                        if norm_col == norm:
                            matched_col = col
                            break

                    if matched_col is not None:
                        val = patient_oars[matched_col]
                        if isinstance(val, str):
                            val = float(val.replace(",", "."))
                        if val <= 0:
                            continue  # OAR not available for this patient

            valid.append(seg_file)

        return valid

    # ------------------------------------------------------------------
    # Mask combination
    # ------------------------------------------------------------------

    def build_combined_mask(
        self,
        seg_files: list,
        reference_image: sitk.Image,
    ) -> np.ndarray:
        """
        Combine individual OAR masks into one integer-labeled volume.

        Filters applied per mask:
          • non-binary files (> MAX_UNIQUE_VALUES unique values)
          • oversized masks  (> MAX_ORGAN_VOLUME_FRACTION of volume)
          • shape mismatches
        """
        ref_shape = sitk.GetArrayFromImage(reference_image).shape
        combined = np.zeros(ref_shape, dtype=np.uint8)

        for seg_file in seg_files:
            oar_name = (
                os.path.basename(seg_file)
                .split("_OAR_")[1]
                .replace(".seg.nrrd", "")
            )

            seg_img = sitk.ReadImage(seg_file)

            seg_array = sitk.GetArrayFromImage(seg_img)

            # Shape safety
            if seg_array.shape != ref_shape:
                print(
                    f"      -> WARNING: shape mismatch for {oar_name} "
                    f"({seg_array.shape} vs {ref_shape}), skipping."
                )
                continue

            # Non-binary filter
            n_unique = len(np.unique(seg_array))
            if n_unique > self.MAX_UNIQUE_VALUES:
                print(
                    f"      -> Skipping non-binary mask "
                    f"({n_unique} unique values): "
                    f"{os.path.basename(seg_file)}"
                )
                continue

            # Volume filter (the "Bed Assassin")
            fg_frac = np.sum(seg_array > 0) / seg_array.size
            if fg_frac > self.MAX_ORGAN_VOLUME_FRACTION:
                print(
                    f"      -> Ignoring massive mask "
                    f"({fg_frac:.1%} of volume): "
                    f"{os.path.basename(seg_file)}"
                )
                continue

            # Assign consistent global OAR ID (no overwrite of existing labels)
            oar_id = self._get_oar_id(oar_name)
            write_mask = (seg_array > 0) & (combined == 0)
            combined[write_mask] = oar_id

        return combined

    # ------------------------------------------------------------------
    # Cropping
    # ------------------------------------------------------------------

    def get_signal_threshold(self, image_array: np.ndarray) -> float:
        """Override in subclass for modality-specific thresholding."""
        return float(image_array.max()) * 0.02

    def crop_to_anatomy(
        self, image_array: np.ndarray, combined_mask: np.ndarray
    ) -> tuple:
        """
        Hybrid Cropping Strategy
        ------------------------
        Z-axis : intersection of segmentation extent and image-signal extent.
                 This avoids including empty slices where the image has no data
                 (fixes the zero-slice problem for registered MRI volumes).
        X / Y  : image-signal extent (tight crop around body / head).
        All axes receive a safety *margin*.
        """
        margin = self.margin
        threshold = self.get_signal_threshold(image_array)
        signal = image_array > threshold

        # --- Z from segmentations ---
        z_seg = np.where(np.any(combined_mask > 0, axis=(1, 2)))[0]

        # --- Z from image signal ---
        z_img = np.where(np.any(signal, axis=(1, 2)))[0]

        # --- X / Y from image signal ---
        y_img = np.where(np.any(signal, axis=(0, 2)))[0]
        x_img = np.where(np.any(signal, axis=(0, 1)))[0]

        if len(z_seg) == 0 or len(y_img) == 0 or len(x_img) == 0:
            print("    Warning: empty extent – returning uncropped.")
            return image_array, combined_mask

        # Start with segmentations
        z_start, z_end = z_seg[0], z_seg[-1]

        # Apply margin
        z_min = max(0, z_start - margin)
        z_max = min(image_array.shape[0] - 1, z_end + margin)
        
        # Clamp to image signal bounds to avoid padding into empty FOV / air
        if len(z_img) > 0:
            z_min = max(z_min, z_img[0])
            z_max = min(z_max, z_img[-1])
            
        if z_min > z_max:
            print("    Warning: Segmentations and image signal do not overlap!")
            return image_array, combined_mask
            
        y_min = max(0, y_img[0] - margin)
        y_max = min(image_array.shape[1] - 1, y_img[-1] + margin)
            
        x_min = max(0, x_img[0] - margin)
        x_max = min(image_array.shape[2] - 1, x_img[-1] + margin)

        cropped_img = image_array[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        cropped_mask = combined_mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]

        return cropped_img, cropped_mask

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    @staticmethod
    def save_dataset(dataset: dict, filename: str):
        """Save a dataset dict as a compressed .npz file."""
        _save_dataset_npz(dataset, filename)

    # ------------------------------------------------------------------
    # Hook – must be implemented by subclass
    # ------------------------------------------------------------------

    def load_image(self, patient_folder: str, patient_idx: str):
        """
        Must return a tuple
            (reference_sitk_image, image_numpy_array)
        or None to skip this patient.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def process_dataset(self, output_filename: str):
        dataset = {}
        patient_folders = sorted(
            [f.path for f in os.scandir(self.data_folder) if f.is_dir()]
        )

        for i, p_folder in enumerate(patient_folders):
            p_idx = os.path.basename(p_folder)
            print(f"\n[{i + 1}/{len(patient_folders)}] Processing {p_idx} ...")

            # 1 – Load the primary image (CT or registered MRI)
            result = self.load_image(p_folder, p_idx)
            if result is None:
                continue
            reference_sitk, image_array = result

            # 2 – Collect and filter segmentation files
            seg_files = self.get_valid_segmentation_files(p_idx, p_folder)
            if not seg_files:
                print(f"  No valid segmentations for {p_idx}, skipping.")
                continue

            # 3 – Build combined mask
            combined_mask = self.build_combined_mask(
                seg_files, reference_sitk
            )

            # Mask out segmentations that extend into the zero-filled void of the registered MRI FOV
            if isinstance(self, MRIProcessor):
                combined_mask[image_array == 0] = 0

            # 4 – Crop
            image_array, combined_mask = self.crop_to_anatomy(
                image_array, combined_mask
            )

            # 5 – Store
            dataset[p_idx] = {
                "image": image_array,
                "segmentations": combined_mask,
            }
            print(f"  -> Success: shape {image_array.shape}")

        if dataset:
            self.save_dataset(dataset, output_filename)
            print(f"\nOAR Mapping: {self.oar_mapping}")
        else:
            print("\nError: no patients were processed!")


# ======================================================================
# CT Processor
# ======================================================================

class CTProcessor(BaseProcessor):
    """Processes CT images – masks are already on the CT grid."""

    def get_signal_threshold(self, image_array: np.ndarray) -> float:
        vmin = float(image_array.min())
        if vmin < -500:
            return -500.0
        return vmin + 1e-3

    def load_image(self, patient_folder, patient_idx):
        ct_paths = glob.glob(os.path.join(patient_folder, "*_IMG_CT.nrrd"))
        if not ct_paths:
            print(f"  CT not found for {patient_idx}, skipping.")
            return None

        ct_sitk = sitk.ReadImage(ct_paths[0])
        ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)

        return ct_sitk, ct_array


# ======================================================================
# MRI Processor
# ======================================================================

class MRIProcessor(BaseProcessor):
    """Processes MRI images – requires registration to CT and mask resampling."""

    def get_signal_threshold(self, image_array: np.ndarray) -> float:
        """MRI: 2 % of max signal separates tissue from background."""
        return float(image_array.max()) * 0.02

    # ------------------------------------------------------------------
    # Rigid registration  (MRI-only)
    # ------------------------------------------------------------------

    def register_and_resample(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
    ) -> sitk.Image:
        """Rigid (Euler 3-D) registration of MRI → CT using Mattes MI."""
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(os.cpu_count())
        
        fixed_f = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_f = sitk.Cast(moving_image, sitk.sitkFloat32)

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.02)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=100,
            gradientMagnitudeTolerance=1e-8,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        init_tx = sitk.CenteredTransformInitializer(
            fixed_f,
            moving_f,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        reg.SetInitialTransform(init_tx, inPlace=False)

        try:
            final_tx = reg.Execute(fixed_f, moving_f)
            print(
                f"  Registration OK  (metric = {reg.GetMetricValue():.4f})"
            )
        except Exception as e1:
            print(f"  Geometry init failed ({e1}), falling back to Identity …")
            reg.SetInitialTransform(sitk.Euler3DTransform(), inPlace=False)
            try:
                final_tx = reg.Execute(fixed_f, moving_f)
                print(
                    f"  Fallback OK  (metric = {reg.GetMetricValue():.4f})"
                )
            except Exception as e2:
                print(f"  Registration failed ({e2}), using identity.")
                final_tx = sitk.Euler3DTransform()

        return sitk.Resample(
            moving_image,
            fixed_image,
            final_tx,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )

    # ------------------------------------------------------------------
    # Image loading  (MRI override)
    # ------------------------------------------------------------------

    def load_image(self, patient_folder, patient_idx):
        ct_paths = glob.glob(os.path.join(patient_folder, "*_IMG_CT.nrrd"))
        mr_paths = glob.glob(os.path.join(patient_folder, "*_IMG_MR_T1.nrrd"))

        if not ct_paths:
            print(f"  CT not found for {patient_idx}, skipping.")
            return None
        if not mr_paths:
            print(f"  MRI not found for {patient_idx}, skipping.")
            return None

        ct_sitk = sitk.ReadImage(ct_paths[0])
        mri_sitk = sitk.ReadImage(mr_paths[0])

        # Register MRI → CT space
        mri_resampled = self.register_and_resample(ct_sitk, mri_sitk)
        mri_array = sitk.GetArrayFromImage(mri_resampled)

        # reference_sitk is the CT (masks live on this grid)
        return ct_sitk, mri_array


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HanSeg → .npz converter (CT or MRI)"
    )
    parser.add_argument(
        "modality", choices=["ct", "mri"], help="Which modality to process"
    )
    parser.add_argument("--data-dir", default="HanSeg/set_1")
    parser.add_argument("--oar-csv", default="HanSeg/set_1/OAR_data.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--margin", type=int, default=15)
    args = parser.parse_args()

    out = args.output or f"HanSeg_{args.modality.upper()}"

    if args.modality == "ct":
        proc = CTProcessor(args.data_dir, args.oar_csv, margin=args.margin)
    else:
        proc = MRIProcessor(args.data_dir, args.oar_csv, margin=args.margin)

    proc.process_dataset(out)
