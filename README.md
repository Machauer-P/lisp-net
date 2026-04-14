# Prompt U-Net

A repository for the development, training, evaluation, and deployment of **Prompt U-Net**, an interactive machine learning model for medical image segmentation based on prompts.

This repository structure emphasizes modularity, clearly separating data loading, training pipelines, model architecture, evaluation baselines, and web deployment.

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd prompt-unet
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   For core ML, data processing, and training, install the default requirements:
   ```bash
   pip install -r requirements.txt
   ```
   If you plan to run evaluations against external benchmark models (like **nnInteractive** or **UniverSeg**), which require PyTorch and specific evaluation libraries, use:
   ```bash
   pip install -r requirements_eval.txt
   ```

## 📂 Project Structure

A detailed overview of the directories and their purpose:

- **`data/`**: Core dataset processing scripts and data loaders.
  - Contains `DataGenerator.py`, `DataLoader.py`, and `DataLoader_npz.py` to efficiently load medical imaging datasets (e.g., `.npz` format).
  - Includes Jupyter notebooks (`visualize_2d_data.ipynb`, `visualize_datagen.ipynb`) to explore data augmentations and data generator outputs.
  - Expected location for `train_data/` and `test_data/` subdirectories.

- **`models/`**: Neural network architecture and related definitions.
  - `prompt_unet.py`: The architecture definition of the **Prompt U-Net** model built with TensorFlow/Keras.
  - `optimizer.py`: Custom optimizers designed for the model's training dynamics.

- **`training/`**: The main hub for training Prompt U-Net.
  - Contains various Jupyter Notebooks (e.g., `p_unet_291.ipynb`) used to orchestrate model training runs.
  - Also acts as the storage area for exported `.keras` models.

- **`utils/`**: Reusable modules for preprocessing, augmentation, and measuring performance.
  - `augmentations.py` & `test_augmentations.py`: Custom augmentation pipelines for robust model training.
  - `preprocessing.py` & `resampling.py`: Code for normalizing shapes and intensities of medical images (CT, MRI).
  - `metrics.py`: Standardized metrics for calculating model performance (like Dice score).
  - `visualization.py`: Helper functions for visualizing model predictions against ground truth labels.

- **`evaluation/`**: Scripts and data configurations for benchmarking tests.
  - `eval_prompt_unet/`: Dedicated evaluations testing Prompt U-Net configurations.
  - `benchmark_models/`: Setup and documentation for external baseline models.
  - `benchmark_nninteractive/` & `benchmark_universeg/`: Specific run scripts and analysis tools against nnInteractive and UniverSeg.

- **`deployment/`**: Web-based deployment tools.
  - Interactive demonstration using TensorFlow.js (`index.html`, `script.js`, `style.css`).
  - `keras_to_tf_js.py`: Script to convert your trained `.keras` architectures into a `.bin`/`.json` format readable by the browser.

## 📖 How to Use

### 1. Training a Model
To train a brand new model or continue training, navigate to the `training/` directory. Open the latest notebook (e.g., `p_unet_291.ipynb`). Ensure your data is correctly populated in `data/train_data/`. The notebooks are designed interactively to configure hyperparameters, instantiate the model via `models/prompt_unet.py`, load data using generators from `data/`, and run training loops.

### 2. Evaluating Models
Once you have trained your Prompt U-Net, you can assess its performance. Ensure you have installed the dependencies via `requirements_eval.txt`. Check the folders inside `evaluation/` for specific run instructions depending on the chosen benchmark to compare Prompt U-Net against leading solutions.

### 3. Deploying to the Web
To demonstrate the model's capabilities:
1. Export your `.keras` model using the `deployment/keras_to_tf_js.py` converter.
2. The UI code inside `deployment/` (HTML, JS, CSS) provides an interactive canvas to draw prompts and see real-time inference using TensorFlow JS.
3. Simply serve the `deployment/` folder with an HTTP server, e.g., `python -m http.server 8000`, and navigate to `localhost:8000` in your web browser.
