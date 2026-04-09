import os
import sys
import argparse
import tensorflow as tf

# Add project root to sys.path to allow imports from data and utils
# root is two levels up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.test_data.ds_handler_2d import save_tf_dataset_2D
from data.DataLoader_npz import DataLoader_npz
from data.DataGenerator import DataGenerator

def gen_save_ds(dg, path, ds_name, offset_val, num_ds=100, max_data_points=100, len_p_set=16):
    """
    Generate and save 2D datasets as TFRecords with integrated offsets.
    
    Args:
        dg: DataGenerator instance.
        path: Directory where datasets will be saved.
        ds_name: Base name for the generated files.
        offset_val: The slice offset to use for prompt-target pairs.
        num_ds: Number of datasets to generate.
        max_data_points: Approximate number of query points per dataset.
        len_p_set: Number of support (prompt) samples per dataset.
    """
    os.makedirs(path, exist_ok=True)
    
    for i in range(num_ds):
        print(f"Generating dataset {i+1}/{num_ds}...")
        
        # Get data points and original offsets from the generator
        # Returns a dataset yielding (x, y, p) tuples and a list of offsets
        ds, offsets = dg.get_data_points_from_one_task(
            max_data_points=max_data_points + len_p_set, 
            offset=offset_val
        )

        # Convert offsets list to a tf.data.Dataset
        offsets_ds = tf.data.Dataset.from_tensor_slices(offsets)

        # Zip dataset elements with their corresponding offsets and flatten the structure
        # Resulting ds_full yields a flat (x, y, p, offset) tuple
        ds_full = tf.data.Dataset.zip((ds, offsets_ds)).map(lambda data, offset: (*data, offset))

        # Split into support and query datasets
        support_ds = ds_full.take(len_p_set).map(lambda x, y, p, offset: (x, y))
        query_ds = ds_full.skip(len_p_set)

        # Save query dataset to TFRecord
        query_filename = f"{i}_{ds_name}.tfrecord"
        save_tf_dataset_2D(query_ds, query_filename, path)
        
        # Save support dataset to TFRecord
        support_filename = f"{i}_{ds_name}_support.tfrecord"
        save_tf_dataset_2D(support_ds, support_filename, path)

    print(f"\nSuccess: Generated {num_ds} datasets in {path}")

def main():
    parser = argparse.ArgumentParser(description="Standalone pipeline for generating 2D benchmark data.")
    parser.add_argument("--input_npz", type=str, nargs='+', default=["data/test_data/HanSeg_MRI.npz"], 
                        help="Path to input .npz file(s) (relative to root or absolute)")
    parser.add_argument("--output_dir", type=str, default="data/test_data/2d/offset_5", 
                        help="Directory to store the generated TFRecords")
    parser.add_argument("--ds_name", type=str, default="HanSeg", help="Base name for the dataset files")
    parser.add_argument("--num_ds", type=int, default=100, help="How many datasets to generate")
    parser.add_argument("--offset", type=int, default=5, help="Slice offset")
    parser.add_argument("--max_points", type=int, default=100, help="Target number of points per dataset")
    parser.add_argument("--len_p_set", type=int, default=16, help="Number of support samples")

    args = parser.parse_args()

    # Resolve paths
    input_paths = []
    for p in args.input_npz:
        full_p = os.path.join(project_root, p) if not os.path.isabs(p) else p
        if not os.path.exists(full_p):
            if os.path.exists(p):
                full_p = os.path.abspath(p)
            else:
                print(f"Error: Could not find input file at {full_p}")
                sys.exit(1)
        input_paths.append(full_p)

    output_path = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    # Initialize components
    print(f"Initializing DataLoader from {input_paths}...")
    dl = DataLoader_npz(input_paths, val_size=0.0)
    dg = DataGenerator(dl)
    
    # Run generation
    gen_save_ds(
        dg, output_path, args.ds_name, args.offset, 
        args.num_ds, args.max_points, args.len_p_set
    )

if __name__ == "__main__":
    main()
