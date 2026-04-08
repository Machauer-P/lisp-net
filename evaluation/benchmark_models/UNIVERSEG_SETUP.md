# Setup and Usage Guide for UniverSeg Benchmarking

This directory contains the `UniverSeg` model used for benchmarking in the `prompt-unet` project.

## How to Set Up UniverSeg

If you are a new user and need to download and install `UniverSeg`, follow these steps:

### 1. Clone the Repository

From the project root:

```powershell
cd evaluation/benchmark_models
git clone https://github.com/JJGO/UniverSeg.git
cd UniverSeg
```

### 2. Create and Activate a Virtual Environment

It is recommended to use a dedicated virtual environment for `UniverSeg`.

**On Windows (PowerShell):**

```powershell
python -m venv venv-univer
.\venv\Scripts\activate
```

**On Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

Once the environment is active, install `UniverSeg` in editable mode:

```powershell
pip install -e .
```

A snapshot of the requirements used during the initial setup can be found in `requirements_snapshot.txt`.

## How to Activate the Environment (Future Use)

Whenever you want to run evaluations using `UniverSeg`, simply navigate to its directory and activate the existing environment:

```powershell
cd evaluation/benchmark_models/UniverSeg

# If using PowerShell
.\venv-univer\Scripts\activate

# If using Git Bash
source venv-univer/Scripts/activate
```

## Troubleshooting

* Ensure you have `python` (3.9+) installed and in your PATH.
* If activation fails in PowerShell, you may need to set the execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
