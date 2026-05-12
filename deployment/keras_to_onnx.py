"""
Export a Prompt U-Net .keras model to ONNX format for ONNX Runtime Web.

Usage:
    python deployment/keras_to_onnx.py --input training/p_unet_332.keras --output deployment/p_unet_332.onnx
    python deployment/keras_to_onnx.py --input training/p_unet_332.keras --output deployment/p_unet_332.onnx --opset 18

Requires: pip install tf2onnx onnx
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.model_loading import load_keras_model

import tensorflow as tf
import tf2onnx


def export_onnx(input_model_path, output_path, opset=17):
    if not os.path.exists(input_model_path):
        print(f"Error: Could not find model at '{input_model_path}'")
        return

    print(f"Loading model from {input_model_path}...")
    model = load_keras_model(input_model_path)

    H, W = model.input_shape[0][1:3]
    print(f"  Inputs:  {[inp.name for inp in model.inputs]}")
    print(f"  Outputs: {[out.name for out in model.outputs]}")
    print(f"  Spatial: {H}x{W}")
    print(f"  Params:  {model.count_params():,}")

    # Explicit input signature for dynamic batch dim
    input_signature = [
        tf.TensorSpec([None, H, W, 1], tf.float32, name='image'),
        tf.TensorSpec([None, H, W, 2], tf.float32, name='prompt'),
    ]

    print(f"\nConverting to ONNX (opset={opset})...")
    _, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=output_path,
    )

    print(f"\nONNX model saved to: {output_path}")
    print(f"  Model size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

    # Validate the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  onnx.checker: OK")

        for inp in onnx_model.graph.input:
            shape = [d.dim_value if d.dim_value else 'N' for d in inp.type.tensor_type.shape.dim]
            print(f"  ONNX input:  {inp.name} {shape}")
        for out in onnx_model.graph.output:
            shape = [d.dim_value if d.dim_value else 'N' for d in out.type.tensor_type.shape.dim]
            print(f"  ONNX output: {out.name} {shape}")
    except ImportError:
        print("  (pip install onnx to validate the exported model)")
    except Exception as e:
        print(f"  onnx.checker warning: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Prompt U-Net .keras to ONNX.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .keras model")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path for output .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()
    export_onnx(args.input, args.output, args.opset)
