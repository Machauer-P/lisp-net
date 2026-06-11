# Training Data Preparation

The model was trained on 208 volumes across 7 datasets, stored as NPZ files in this directory. Before running `training/train_332.py`, you must populate this folder with the required NPZ files.

## Dataset Status

| Dataset | Conversion Script | Status |
|---------|------------------|--------|
| BraTS-GLI | `brats_to_npz.py` | ✅ Portable — works with publicly downloaded BraTS 2024 data |
| BraTS-MEN-RT | `brats_to_npz.py` | ✅ Portable — works with publicly downloaded BraTS 2024 data |
| TopCoW MR | `top_cow_to_npz.py` | ✅ Portable — works with publicly downloaded TopCoW 2024 data |
| TopCoW CT | `top_cow_to_npz.py` | ✅ Portable — works with publicly downloaded TopCoW 2024 data |
| NAKO | `nako_to_npz.py` | ⚠️ Uses `dpx_loader.py` — institution-specific infrastructure |
| TotalSegmentator | `total_seg_to_npz.py` | ⚠️ Uses `dpx_loader.py` — institution-specific infrastructure |
| MSD (Medical Decathlon) | `med_dec_to_npz.py` | ⚠️ Uses `dpx_loader.py` — institution-specific infrastructure |

## Portable Datasets (BraTS, TopCoW)

These work out of the box. Download the raw data from the challenge websites:

```bash
# BraTS 2024 (two sub-challenges)
python data/train_data/brats_to_npz.py gli     --data-dir <path_to_BraTS-GLI>     --output BraTS_GLI_t1c
python data/train_data/brats_to_npz.py men_rt  --data-dir <path_to_BraTS-MEN-RT>  --output BraTS_MEN_RT

# TopCoW 2024 (two modalities)
python data/train_data/top_cow_to_npz.py ct --data-dir <path_to_TopCoW> --output TopCoW_CT
python data/train_data/top_cow_to_npz.py mr --data-dir <path_to_TopCoW> --output TopCoW_MR
```

## Institution-Specific Datasets (NAKO, TotalSegmentator, MSD)

The existing `nako_to_npz.py`, `total_seg_to_npz.py`, and `med_dec_to_npz.py` scripts rely on `dpx_loader.py`, which connects to internal institutional infrastructure (environment variables `DPXROOT`, `DPXproject`, and the proprietary `DPX_core` / `patchwork_dev` libraries). This infrastructure is **not publicly available**.

To use the publicly available versions of these datasets, you need to write custom `_to_npz.py` conversion scripts. Use `brats_to_npz.py` and `top_cow_to_npz.py` as templates.

### Writing a Custom `_to_npz.py` Script

The only contract is: your script must produce an NPZ file via `ds_handler.save_dataset()` (from `data/test_data/ds_handler.py`) with the following structure:

```python
dataset = {
    "patient_001": {
        "image":         np.ndarray,        # 3-D volume (Z, Y, X), isotropically resampled
        "segmentations": np.ndarray,        # 3-D label volume (Z, Y, X), same shape as image
        "modality":      "CT" | "MRI",      # imaging modality tag
    },
    "patient_002": { ... },
    ...
}

from data.test_data.ds_handler import save_dataset
save_dataset(dataset, "output_filename.npz")
```

Key requirements:
- **Image and label volumes must be isotropically resampled** (use `utils.resampling.resample_isotropic`).
- **Cropping**: Apply `utils.cropping.crop_to_anatomy` to remove empty background regions (reduces file size and speeds up training).
- See `brats_to_npz.py` for a complete, self-contained example of the conversion pipeline (loading, resampling, cropping, saving).

### Available Utility Functions

| Utility | Location | Purpose |
|---------|----------|---------|
| `resample_isotropic` | `utils/resampling.py` | Resample volumes to isotropic spacing |
| `crop_to_anatomy` | `utils/cropping.py` | Crop zero-background regions |
| `load_medical_image` | `utils/itk_utils.py` | Load NIfTI / NRRD / DICOM with SimpleITK |
| `get_voxel_sizes` | `utils/itk_utils.py` | Read voxel spacing from image header |
| `save_dataset` | `data/test_data/ds_handler.py` | Serialize dataset dict to .npz |

### Expected NPZ Files

The training script `training/train_332.py` expects these files in `data/train_data/`:

```
data/train_data/
├── nako_combined.npz
├── total_seg_combined.npz
├── msd_combined.npz
├── brats_gli.npz
├── brats_men_rt.npz
├── TopCoW_MR.npz
└── TopCoW_CT.npz
```
