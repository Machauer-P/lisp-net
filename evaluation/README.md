# Model Evaluation & Benchmarking

This directory contains the tools and scripts required to assess the performance of **LISP-Net** and compare it against established baseline models (**UniverSeg** and **nnInteractive**).

## 🚀 Getting Started

To run the evaluation scripts, you must install the evaluation dependencies, populate the benchmark model repositories, and prepare the test datasets.

### 1. Install Dependencies
Evaluations require PyTorch and specific numerical libraries not included in the core training requirements.
```bash
pip install -r requirements_eval.txt
```

### 2. Populate Benchmark Models
The benchmark models reside in the `evaluation/benchmark_models/` directory. If this folder is empty, clone the following repositories into it:

```bash
# Navigate to the benchmark models directory
cd evaluation/benchmark_models

# Clone UniverSeg
git clone https://github.com/JJGO/UniverSeg.git UniverSeg

# Clone nnInteractive
git clone https://github.com/MIC-DKFZ/nnInteractive.git nnInteractive
```

### 3. Prepare Test Datasets

The benchmark scripts expect NPZ bundles in `data/test_data/`. You must download the raw datasets and convert them using the provided scripts:

**2D Benchmark datasets:** FLARE, HanSeg, HCCTase, SegRap2023, TotalSegmentator MRI
```bash
# Convert each raw dataset to NPZ format
python data/test_data/flare_to_npz.py --data_path <path_to_flare>
python data/test_data/han_seg_to_npz.py --data_path <path_to_hanseg>
python data/test_data/hcctase_to_npz.py --data_path <path_to_hcctase>
python data/test_data/segrap_to_npz.py --data_path <path_to_segrap>
python data/test_data/totalseg_to_npz.py --data_path <path_to_totalseg_mri>

# Generate 2D offset bundles (offset_5 and offset_12)
python data/test_data/generate_2d_test_data.py
```

**3D Benchmark datasets:** The same 5 datasets as 2D (FLARE, HanSeg, HCCTase, SegRap2023, TotalSegmentator MRI), used as raw 3D NPZ volumes rather than 2D offset bundles. After running the conversion scripts above, the resulting `.npz` files are passed directly to `benchmark_3d.py`.

**3D OOD (Generalization) datasets:** Couinaud, Mouse — for the out-of-distribution generalization benchmark only.
```bash
python data/test_data/couinaud_to_npz.py --data_path <path_to_couinaud>
python data/test_data/mouse_to_npz.py --data_path <path_to_mouse>
```

> **Note:** The raw datasets must be obtained from their respective sources. See the individual `*_to_npz.py` scripts for dataset-specific download instructions and expected directory structures.

### 4. LISP-Net Model (Auto-Download or Local)

By default, the benchmark scripts **automatically download** the pre-trained LISP-Net weights from [Hugging Face](https://huggingface.co/Machauer-P/lisp-net). No manual model download is required.

To use a **self-trained model** instead, pass the path to your local `.keras` file:
```bash
# 2D benchmark with custom model
python evaluation/benchmark_universeg/benchmark_2d.py \
    --model prompt_unet \
    --data_path data/test_data/2d/offset_5 \
    --p_unet_model training/p_unet_332.keras

# 3D benchmark with custom model
python evaluation/benchmark_nninteractive/benchmark_3d.py \
    --npz_paths data/test_data/<your_volume>.npz \
    --p_unet_model training/p_unet_332.keras
```

---

## 📂 Directory Structure

- **`benchmark_models/`**: Contains the code for external baseline models (UniverSeg, nnInteractive).
- **`benchmark_universeg/`**: 2D benchmark pipeline and pre-computed results comparing LISP-Net with UniverSeg.
- **`benchmark_nninteractive/`**: 3D benchmark pipeline and pre-computed results comparing LISP-Net with nnInteractive.

---

## 📓 Pre-Computed Results (Notebooks)

The following notebooks contain **all results reported in the paper** — open them to inspect figures, tables, and raw data without re-running the full pipeline:

| Benchmark | Notebook |
|-----------|----------|
| 2D (LISP-Net vs UniverSeg) | `evaluation/benchmark_universeg/benchmark_2d_and_results.ipynb` |
| 2D Generalization (OOD) | `evaluation/benchmark_universeg/generalization_2d_results.ipynb` |
| 3D (LISP-Net vs nnInteractive) | `evaluation/benchmark_nninteractive/benchmark_3d_results.ipynb` |
| 3D Generalization (OOD) | `evaluation/benchmark_nninteractive/generalization_3d_results.ipynb` |

To **re-run** benchmarks with your own model or parameter changes, use the Python scripts directly:

```bash
# 2D benchmark
python evaluation/benchmark_universeg/benchmark_2d.py \
    --model prompt_unet \
    --data_path data/test_data/2d/offset_5

# 3D benchmark
python evaluation/benchmark_nninteractive/benchmark_3d.py \
    --npz_paths data/test_data/<your_volume>.npz \
    --p_unet_model training/p_unet_332.keras \
    --nn_model_dir evaluation/benchmark_models/nnInteractive
```

For complexity benchmarks see:
- `evaluation/benchmark_universeg/complexity_universeg.ipynb`
- `evaluation/benchmark_nninteractive/inference_speed.ipynb`
- `evaluation/benchmark_nninteractive/p_unet_memory.ipynb`