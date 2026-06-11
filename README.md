# LISP-Net (Lightweight In-Context Slice Propagator Network)

[![Live Demo](https://img.shields.io/badge/Demo-Launch%20in%20Browser-green?style=for-the-badge)](https://www.nora-imaging.com/)
[![Paper](https://img.shields.io/badge/Paper-Preprint-blue?style=for-the-badge)](docs/LISP_NET_PREPRINT.pdf)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Hugging%20Face-orange?style=for-the-badge)](https://huggingface.co/Machauer-P/lisp-net)

LISP-Net is a lightweight, purely convolutional framework for interactive volumetric medical image segmentation. A single dense 2D prompt — a reference image paired with a full segmentation mask — defines both the target structure and the desired boundary style, propagating the user's annotation intent through within-volume visual correspondence. Pre-trained weights are available on [Hugging Face](https://huggingface.co/Machauer-P/lisp-net) as `.keras` and `.onnx`.

**Key Advantages:**
1. **Full Client-Side Inference:** Medical data never leaves the local machine — no server round-trips, no cloud dependency.
2. **Fast & Low Hardware Requirements:** Peak GPU memory of 362 MB, per-slice latency of ~14 ms (GPU) / ~150 ms (CPU). Runs on consumer-grade hardware.
3. **Prompt-Driven, Not Prior-Driven:** The model follows the user's annotation intent rather than imposing memorized anatomical priors. This makes it robust to out-of-distribution data: e.g. an MRI-only variant retains accuracy on CT, and a head-only variant segments body anatomy.

### Watch the Demo or Try it Yourself
- **YouTube Demo:** [Watch our demo video](https://youtu.be/SafGK6U0nDI)
- **Interactive Demo:** Use it yourself at [Nora Imaging](https://www.nora-imaging.com/)
  - Read the [Nora Imaging Documentation](https://reisertm.github.io/noradoc/chapters/segmentation-assistant-lisp-net.html) first

> **⚠️ Research Prototype:** The Nora Imaging demo is a research prototype tested on a limited set of platforms and browsers. For validated clinical deployment, practitioners should use the [open model weights](https://huggingface.co/Machauer-P/lisp-net) and source code to integrate LISP-Net into their own validated infrastructure.

<br>
<img src="docs/visual_abstract/visual_abstract.png" style="width: 100%;" alt="LISP-Net Visual Abstract">

*Overview of segmentation paradigms.* **(Upper left)** *Classic supervised ML trains a fixed model per task and fails on unseen tasks.* **(Lower left)** *Standard in-context learning uses prompts to define new tasks but offers no correction mechanism.* **(Right)** *LISP-Net combines interactive ICL with a single dense 2D prompt. Within-volume visual correspondence propagates the user's annotation with strong alignment to the user's intent and high accuracy over medium-range offsets. Structural drift is handled automatically by SSF or manually by the user, both of which refresh the prompt context.*
<br>

---

## 📦 Quickest Path to Use This Repo

You don't need to train a model from scratch. The fastest way to get value:

1. **Browse pre-computed results** — Open the [results notebooks](#3-evaluating-models) to inspect all paper figures and tables without running a single line of code.
2. **Grab the model weights** — Download pre-trained `.keras` or `.onnx` from [Hugging Face](https://huggingface.co/Machauer-P/lisp-net) (auto-downloaded by the inference code, or grab them manually).
3. **Drop into your pipeline** — Take the tiling, batching, and inference logic from `inference/predictor.py` and `inference/tiling.py`, normalize inputs with `universal_normalization()` from `utils/preprocessing.py`, inject the model, and integrate LISP-Net into your own clinical workflow.

> The codebase uses the working name `prompt_unet` for internal identifiers (class names, filenames) — this is the same LISP-Net model.

```python
from inference.predictor import PromptUNetPredictor

# Auto-downloads from Hugging Face (no local model file needed):
predictor = PromptUNetPredictor()                       # default: Machauer-P/lisp-net
# Or pass a local .keras file:
predictor = PromptUNetPredictor("training/p_unet_332.keras")

mask = predictor.predict(query_image, prompt)           # any resolution — tiling is automatic
```

> **⚠️ Normalization:** `PromptUNetPredictor` expects **already-normalized** inputs (z-score clipped to `[-5, 5]`). For raw medical volumes, use `VolumeInference` from `inference/inference_volume.py` instead — it applies modality-aware normalization automatically (`universal_normalization()` from `utils/preprocessing.py`). Passing raw intensities directly will silently produce garbage predictions.

## 💡 Core Innovation & Features

LISP-Net is a **purely convolutional framework** for interactive volumetric segmentation. Instead of sparse clicks, it conditions on a single dense 2D anchor prompt and propagates the annotation intent through within-volume visual correspondence rather than memorized anatomical priors. In 2D benchmarks it outperforms UniverSeg by 23.73% at mid-range offsets; in 3D it exceeds nnInteractive by 8.52% in volumetric Dice under simulated user interaction, with further advantages on out-of-distribution data.

**Key Contributions:**
- **Asymmetrical Dual-Encoder:** A heavy Prompt Encoder extracts structural semantics from a dense 2D prompt (reference image + full mask). Multi-resolution SE channel-attention fuses these features into the Query Encoder at each stage.
- **Efficient & Lightweight:** ~28M parameters, adaptive tiling, peak GPU memory of 164–362 MB, per-slice latency of ~14 ms (GPU) / ~150 ms (CPU). ONNX export enables browser-based inference for research demos.
- **Self-Supervised Feedback (SSF):** Automatically detects structural drift during 3D propagation via confidence monitoring and refreshes the prompt context without ground-truth annotations.
- **Interactive Feedback (IFL):** A clinician corrects a missegmented slice; it becomes a fresh dense prompt to update all subsequent predictions.
- **Out-of-Distribution Robustness:** An MRI-only variant retains accuracy on CT; a head-only variant segments body anatomy. The model follows the prompt, not the training prior.

<br>
<img src="docs/p_unet_architecture.png" style="width: 100%;" alt="LISP-Net Architecture">
<br>

---

## 🛠️ Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/Machauer-P/lisp-net
cd lisp-net
```

### 2. Set up a virtual environment:
```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install dependencies:
For core ML, training, and inference:
```bash
pip install -r requirements.txt
```
For evaluation against external benchmarks (nnInteractive, UniverSeg — requires PyTorch):
```bash
pip install -r requirements_eval.txt
```

---

## 📂 Project Structure

```
.
├── data/                       # Data loading, preprocessing & dataset generation
│   ├── DataLoader_npz.py       # Loads NPZ files into RAM, provides dataset dict to DataGenerator
│   ├── DataGenerator.py        # Samples patches, z-score normalization, label-guided 128×128 cropping
│   ├── train_data/             # Scripts to convert raw medical data → training NPZ files (see README.md inside)
│   └── test_data/              # Scripts to convert raw medical data → test NPZ files
│
├── training/                   # Model definitions & training scripts
│   ├── prompt_unet.py          # Current model architecture definition
│   ├── train_332.py            # Final benchmark training script (v332)
│   ├── optimizer.py            # WarmupFlatCosineDecay LR schedule
│   ├── augmentations.py        # Photometric, geometric & morphological augmentations
│   └── CHANGELOG.md            # Full model version history (v21 → v340)
│
├── inference/                  # Prediction pipeline & post-processing
│   ├── predictor.py            # PromptUNetPredictor (direct 128×128 + tiling paths)
│   ├── tiling.py               # TiledInference for arbitrary-resolution slices
│   ├── inference_volume.py     # VolumeInference, SSF, and IFL orchestration
│   ├── ssf.py                  # Self-Supervised Feedback strategies
│   └── tune_ssf.py             # SSF hyperparameter tuning on training data
│
├── evaluation/                 # Benchmarks against UniverSeg (2D) & nnInteractive (3D)
│   ├── README.md               # Evaluation setup & instructions
│   ├── benchmark_universeg/    # 2D comparison pipeline & results
│   └── benchmark_nninteractive/# 3D comparison pipeline & results
│
├── deployment/                 # ONNX export & basic integration test
│   ├── keras_to_onnx.py        # Export .keras → ONNX
│   ├── benchmark.html          # ONNX vs. TF.js correctness check (testing only)
│   ├── index.html              # Minimal browser demo (testing only — not a medical viewer)
│   ├── script.js               # ONNX inference engine & canvas interaction
│   └── style.css               # Demo UI styling
│
├── utils/                      # Shared utilities
│   ├── preprocessing.py        # Normalization, resampling helpers
│   ├── cropping.py             # Patch extraction utilities
│   ├── metrics.py              # Dice, IoU, and other segmentation metrics
│   ├── visualization.py        # Plotting and result visualization
│   ├── model_loading.py        # Model loading with Keras 3 serialization workarounds
│   └── resampling.py           # Volume resampling utilities
│
├── docs/                       # Architecture diagrams and preprint
├── requirements.txt            # Core dependencies (TF 2.21, Keras 3, NumPy, etc.)
└── requirements_eval.txt       # Core dependencies + Evaluation dependencies (PyTorch, nnUNet, UniverSeg)
```

---

## 📖 How to Use the Code

### 1. Training a Model

To train the final benchmark model (v332):

```bash
python training/train_332.py
```

The training script uses:
- **Architecture:** `training/prompt_unet.py` — dual-encoder U-Net with SE attention on prompt skip connections, pure Conv2D, filter schedule `[48, 96, 192, 256, 384]` (~28M params).
- **Data pipeline:** `data/DataLoader_npz.py` loads NPZ files into RAM and provides the dataset dict. `data/DataGenerator.py` takes a DataLoader as input and samples prompt-query pairs with z-score normalization and label-guided 128×128 patch cropping. Augmentations are applied separately via `training/augmentations.py`.
- **LR schedule:** `WarmupFlatCosineDecay` (50 ep warmup → 1500 ep flat → cosine decay to epoch 4000).
- **Loss:** Binary cross-entropy.
- **Training data:** 208 volumes across 7 datasets (NAKO, TotalSegmentator, MSD, BraTS-GLI, BraTS-MEN-RT, TopCoW MR, TopCoW CT), stored as NPZ files in `data/train_data/`.

> **⚠️ Before training:** You must populate `data/train_data/` with NPZ files. See [data/train_data/README.md](data/train_data/README.md) for details.

Training loops are custom (not `model.fit()`), using `tf.GradientTape` with periodic MLflow logging.

### 2. Running Inference

Use the modules in `inference/` — the model auto-downloads from Hugging Face, or you can pass a local `.keras` file:

```python
from inference.predictor import PromptUNetPredictor

# Auto-download from Hugging Face (default):
predictor = PromptUNetPredictor()

# Or load a specific HF repo or local file:
predictor = PromptUNetPredictor("Machauer-P/lisp-net")       # explicit HF repo
predictor = PromptUNetPredictor("training/p_unet_332.keras")  # local .keras file

mask = predictor.predict(query_image, prompt)
```

Two prediction paths are available:
- **Direct 128×128:** Fast batched forward pass for standard-resolution inputs.
- **Tiling:** Adaptive tiling via `TiledInference` for arbitrary-resolution slices (matches the JS implementation in `deployment/script.js`).

For 3D volumes, use `inference/inference_volume.py` which orchestrates slice-by-slice prediction with optional SSF (Self-Supervised Feedback) and IFL (Interactive Feedback Loop).

SSF strategies in `inference/ssf.py` include `RelativeSSIMStrategy`, `MaskDiceStrategy`, and `ConfidenceDropStrategy`.

### 3. Evaluating Models

Ensure evaluation dependencies are installed (`pip install -r requirements_eval.txt`). See [evaluation/README.md](evaluation/README.md) for benchmark model and test dataset setup instructions.

Pre-computed results for the numbers reported in the paper are available in the following notebooks:

| Benchmark | Notebook |
|-----------|----------|
| 2D (LISP-Net vs UniverSeg) | `evaluation/benchmark_universeg/benchmark_2d_and_results.ipynb` |
| 2D Generalization (OOD) | `evaluation/benchmark_universeg/generalization_2d_results.ipynb` |
| 3D (LISP-Net vs nnInteractive) | `evaluation/benchmark_nninteractive/benchmark_3d_results.ipynb` |
| 3D Generalization (OOD) | `evaluation/benchmark_nninteractive/generalization_3d_results.ipynb` |

To re-run benchmarks with your own model or parameter changes, follow the step-by-step instructions in [evaluation/README.md](evaluation/README.md).

Complexity benchmarks (parameter count, FLOPs, speed, memory) are in `evaluation/benchmark_universeg/complexity_universeg.ipynb`, `evaluation/benchmark_nninteractive/inference_speed.ipynb`, and `evaluation/benchmark_nninteractive/p_unet_memory.ipynb`.

### 4. Exporting for Deployment

The `deployment/` folder provides scripts to export a trained `.keras` model to ONNX. The accompanying HTML/JS/CSS files are a minimal test harness to verify the export works correctly — they are **not** a medical image viewer.

```bash
python deployment/keras_to_onnx.py training/p_unet_332.keras
```

To verify the export locally:
```bash
cd deployment && python -m http.server 8000
```

---

## ⚠️ Known Issues

- **Keras 3 serialization bug:** `.keras` files saved by TF 2.21 contain `renorm`/`quantization_config` keys in layer configs that Keras 3 rejects on load. Fixed in `inference/predictor.py` via JSON config patching.
