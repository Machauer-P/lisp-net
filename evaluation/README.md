# Model Evaluation & Benchmarking

This directory contains the tools and scripts required to assess the performance of **Prompt U-Net** and compare it against established baseline models (**UniverSeg** and **nnInteractive**).

## 🚀 Getting Started

To run the evaluation scripts, you must first install the specific evaluation dependencies and populate the benchmark model repositories.

### 1. Install Dependencies
Evaluations require PyTorch and specific numerical libraries not included in the core training requirements.
```bash
pip install -r requirements_eval.txt
```

### 2. Populate Benchmark Models
The benchmark models reside in the `evaluation/benchmark_models/` directory. If this folder is empty, you must clone the following repositories into it:

```bash
# Navigate to the benchmark models directory
cd evaluation/benchmark_models

# Clone UniverSeg
git clone https://github.com/JJGO/UniverSeg.git UniverSeg

# Clone nnInteractive
git clone https://github.com/MIC-DKFZ/nnInteractive.git nnInteractive
```

---

## 📂 Directory Structure

- **`benchmark_models/`**: Contains the code for external baseline models.
- **`benchmark_universeg/`**: Pipeline and results for comparing Prompt U-Net with UniverSeg on 2D datasets.
- **`benchmark_nninteractive/`**: Evaluation notebooks for 3D interactive segmentation using nnInteractive.
- **`eval_prompt_unet/`**: Scripts and notebooks focused on evaluating different versions of Prompt U-Net.
- **`complexity/`**: Notebooks for calculating model parameters, MACs (FLOPs), and inference latency.

---

## 🔬 Running Evaluations

Each subdirectory contains its own set of notebooks or scripts. For example, to run the 2D benchmark pipeline:

```bash
python evaluation/benchmark_universeg/eval_pipeline_2d.py --data_path data/test_data/2d/offset_5
```

Or use the .ipynb notebooks in the respective folders for running the evaluations and analyzing the results.