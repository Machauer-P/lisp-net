import sys
from unittest.mock import MagicMock
import numpy as np

# Mock missing optional dependency that tensorflowjs tries to import
sys.modules["tensorflow_decision_forests"] = MagicMock()

# Fix NumPy 2.0+ removed aliases that tensorflowjs still uses
if not hasattr(np, "object"): np.object = object
if not hasattr(np, "bool"):   np.bool = bool
if not hasattr(np, "float"):  np.float = float

import tensorflow as tf
import tensorflowjs as tfjs
import argparse
import os
import shutil
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.model_loading import load_keras_model


def export_model(input_model_path, output_tfjs_dir, fmt="graph"):
    if not os.path.exists(input_model_path):
        print(f"Error: Could not find model at '{input_model_path}'")
        return

    print(f"Loading model from {input_model_path}...")
    model = load_keras_model(input_model_path)

    if fmt == "graph":
        # GraphModel export via SavedModel bridge
        # This avoids Keras 3 JSON incompatibilities with TFJS LayersModel
        temp_dir = tempfile.mkdtemp()
        try:
            print(f"Step 1: Saving temporary SavedModel to {temp_dir}...")
            tf.saved_model.save(model, temp_dir)

            print(f"Step 2: Converting SavedModel → TFJS GraphModel in '{output_tfjs_dir}/'...")
            if os.path.exists(output_tfjs_dir):
                shutil.rmtree(output_tfjs_dir)
            tfjs.converters.convert_tf_saved_model(temp_dir, output_tfjs_dir)

            print("Export complete! GraphModel files are ready.")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    else:
        # Legacy LayersModel export (may not work with Keras 3 in browser)
        print(f"Exporting LayersModel to '{output_tfjs_dir}/'...")
        tfjs.converters.save_keras_model(model, output_tfjs_dir)
        print("Export complete! LayersModel files are ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Keras model to TensorFlow.js format.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .keras model file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Directory to save output tfjs model")
    parser.add_argument("--format", "-f", type=str, default="graph", choices=["graph", "layers"],
                        help="Output format: 'graph' (recommended) or 'layers'")

    args = parser.parse_args()
    export_model(args.input, args.output, args.format)