# Setup and Usage Guide for nnInteractive Benchmarking

This directory contains the `nnInteractive` model used for benchmarking in the `prompt-unet` project.

## How to Set Up nnInteractive

If you are a new user and need to download and install `nnInteractive`, follow these steps:

### 1. Clone the Repository
From the project root:
```powershell
cd evaluation/benchmark_models
git clone https://github.com/MIC-DKFZ/nnInteractive
cd nnInteractive
```

### 2. Create and Activate a Virtual Environment
It is recommended to use a dedicated virtual environment for `nnInteractive`.

**On Windows (PowerShell):**
```powershell
python -m venv venv-nn
.\venv-nn\Scripts\activate
```

**On Windows (Git Bash):**
```bash
python -m venv venv-nn
source venv-nn/Scripts/activate
```

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
Once the environment is active, install `nnInteractive` in editable mode:
```powershell
pip install -e .
```
A snapshot of the requirements used during the initial setup can be found in `requirements_snapshot.txt`.

## How to Activate the Environment (Future Use)

Whenever you want to run evaluations using `nnInteractive`, simply navigate to its directory and activate the existing environment:

```powershell
cd evaluation/benchmark_models/nnInteractive

# If using PowerShell
.\venv-nn\Scripts\activate

# If using Git Bash
source venv-nn/Scripts/activate
```

## Troubleshooting
- Ensure you have `python` (3.9+) installed and in your PATH.
- If activation fails in PowerShell, you may need to set the execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.
