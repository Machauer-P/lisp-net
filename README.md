# LISP-Net (Lightweight In-Context Slice Propagator Network)

> **Working name / formerly LISP-Net.** Renamed to avoid a naming conflict with an existing LISP-Net; the codebase still uses the original `prompt_unet` identifiers internally.

[![Live Demo](https://img.shields.io/badge/Demo-Launch%20in%20Browser-green?style=for-the-badge)](https://www.nora-imaging.com/)
[![Status](https://img.shields.io/badge/Paper-Preprint-blue?style=for-the-badge)](docs/p_unet_preprint_outdated.pdf)

> **"A leap towards generalizable, lightweight, and user-controllable AI in clinical workflows."**

Welcome to the code repository for **LISP-Net**, an interactive machine learning model for medical image segmentation based on prompts. **Here, you can build, train, and evaluate the model yourself.**

If you simply want to try the model **without any installation or setup problems**, our interactive demo can be used directly in your browser via [Nora Imaging](https://www.nora-imaging.com/).

---

## 🚀 Interactive Demo & Clinical Accessibility

To bridge the gap between research and clinical application, the model is highly optimized and deployed using TensorFlow.js. We demonstrate that, compared to resource-intensive models, high-fidelity segmentation can be achieved locally on standard consumer hardware.

**Why this matters for Clinicians/Researchers:**
1. **Zero-Setup:** No Python/Docker/GPU drivers needed. Works instantly in any browser via Nora Imaging.
2. **Data Privacy:** Full **client-side inference**. Medical data never leaves the local machine.
3. **In-context learning:** Perfect for real-time interaction during imaging or screening.

### Watch the Demo or Try it Yourself
- **YouTube Demo:** [Watch our demo video](https://www.youtube.com/watch?v=DFkN3o8yA4w)
- **Interactive Demo:** Use it yourself on your device inside [Nora Imaging](https://www.nora-imaging.com/)
  - **Currently uses LISP-Net V21, which is intended solely for MRI segmentation and is slower than the latest version.**
  - [Nora Imaging Documentation](https://www.nora-imaging.org/doc)
  1. Press 'M' to memorize the segmentation you made.
  2. Press 'N' on another slice of the same axis to create a segmentation.
  3. Proceed if the result meets your expectations. If not, edit it and memorize it.

---

## 💡 Scientific Core Innovation & Features

LISP-Net transforms static segmentation into an **interactive, context-aware process**. Unlike "black-box" models, it leverages **In-Context Learning** to adapt to unseen anatomical structures using minimal data, while outperforming UniverSeg and rivaling the performance of nnInteractive, all with significantly lower computational complexity and memory footprint.

**Features:**
- **Dual-Encoder Architecture:** Simultaneously processes a medical image and a 2D user-provided prompt, with a dedicated conditioning mechanism.
- **In-Context Learning:** Enables rapid adaptation to new tasks without retraining.
- **Self-Supervised Feedback (SSF):** Automatically ensures volumetric consistency. The model uses its own predictions from adjacent slices as internal "context" to refine the current segmentation without human intervention *(not yet in Browser Demo)*.
- **Interactive Feedback (IF):** Enables "Human-in-the-loop" refinement. A clinician can provide a manual correction on a missegmented area, which is instantly used to update and improve future masks.
- **Data Efficiency:** Rivaling top-tier baselines like nnInteractive while requiring significantly less data.

<br>
<img src="docs/p_unet_architecture.png" style="width: 100%;" alt="LISP-Net Architecture">
<br>

---

## 🛠️ Code Repository & Getting Started

This repository structure emphasizes modularity, clearly separating data loading, training pipelines, model architecture, evaluation baselines, and web deployment.

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
For core ML, data processing, and training, install the default requirements:
```bash
pip install -r requirements.txt
```
If you plan to run evaluations against external benchmark models (like **nnInteractive** or **UniverSeg**), which require PyTorch and specific evaluation libraries, use:
```bash
pip install -r requirements_eval.txt
```

---

## 📂 Project Structure

- **`data/`**: Core dataset processing scripts and data loaders (e.g., `DataGenerator.py`, `DataLoader_npz.py`). Contains Jupyter notebooks to explore data augmentations, visualize datasets, and scripts to process raw medical data into efficient `.npz` records. Use this folder to manage and prepare all training and evaluation data.
- **`inference/`**: Modules for running predictions with trained models. Includes the core inference classes (e.g., `inference_volume.py`), logic for adaptive spatial tiling (`tiling.py`), and the Self-Supervised Feedback mechanism (`ssf.py`) that intelligently enforces volumetric consistency across 3D image slices without human intervention.
- **`training/`**: The main hub for defining and training LISP-Net. Contains all neural network architecture variants (`prompt_unet_*.py`), custom optimizers, logging configurations, and Jupyter Notebooks (e.g., `p_unet_*.ipynb`) used to run historical and active training experiments. Trained `.keras` model weights are built and stored here.
- **`utils/`**: Reusable helper modules that handle common operations across the codebase. Includes utilities for image processing, specialized data augmentations, visualization routines, and measuring performance metrics.
- **`evaluation/`**: Scripts, pipeline tools, and Jupyter notebooks explicitly designed for robust benchmarking. Use this to analyze the performance of LISP-Net against competing baselines (such as nnInteractive and UniverSeg) and generate comparative statistics across anatomical tasks.
- **`deployment/`**: Web-based deployment tools. Contains scripts (e.g., `keras_to_tf_js.py`) to convert trained `.keras` format models into TensorFlow.js graph models, alongside basic HTML/JS/CSS assets for an interactive browser-based testing demonstration.
- **`docs/`**: Relevant documents such as architectural diagrams, preprints, and research to provide a theoretical understanding of the LISP-Net model and its underlying methodology.

---

## 📖 How to Use the Code

### 1. Training a Model
To train a brand new model or continue training, navigate to the `training/` directory. Open the latest notebook (e.g., `p_unet_292.ipynb`). Ensure your data is correctly populated in `data/train_data/`. The notebooks are designed interactively to configure hyperparameters, instantiate the model via `models/prompt_unet.py`, load data using generators from `data/`, and run training loops.

### 2. Evaluating Models
Once you have trained your LISP-Net, you can assess its performance. Ensure you have installed the dependencies via `requirements_eval.txt`. Check the folders inside `evaluation/` for specific run instructions depending on the chosen benchmark.

### 3. Model Inference, Exporting & Deployment
Once trained, the `.keras` model can be utilized in several ways:
- **Python Inference:** Use the modules inside the `inference/` folder (such as `inference_volume.py`) to run predictions and apply the Self-Supervised Feedback (SSF) mechanism directly from Python. This is the primary method for predicting on 3D medical volumes.
- **Exporting for the Web:** You can export your trained `.keras` model to a TensorFlow.js format using the `deployment/keras_to_tf_js.py` converter tool.
- **Web Deployment (Demo Only):** The `deployment/` folder includes UI code (HTML, JS, CSS) to serve an interactive canvas using the exported model via a local HTTP server (`python -m http.server 8000`). *Note: This local deployment is strictly for testing the TensorFlow.js integration and basic drawing mechanics. It is not useful for real-world application or general usage, as it lacks a proper 3D medical image viewer and interactive context update mechanism.*

---

## 🔬 Technical Documentation & Publication

For an in-depth discussion, please refer to our preprint on version 21 of the LISP-Net:

**[Read Preprint](docs/p_unet_preprint_outdated.pdf)**

> **Note on Project Evolution:** 
> This codebase is under active development. While the preprint provides the foundational scientific framework, the current implementation has evolved further. The future publication will include architectural refinements and further research, for example into computational complexity and memory footprint.
