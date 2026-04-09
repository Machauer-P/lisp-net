import os
import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
Augmentation Testing Script for PromptUNet.
Generates visualizations and statistics for both CT and MRI data.

Usage Examples:
    # Run with default probabilities (1.0) and default output (utils/results)
    python utils/test_augmentations.py

    # Run with a global probability of 0.5 for all augmentations
    python utils/test_augmentations.py --prob 0.5

    # Run with custom output directory and specific probability overrides
    python utils/test_augmentations.py --output_dir utils/results/noise
        --prob_photo 0.0 
        --prob_gamma 0.0 
        --prob_noise 1.0 
        --prob_independent_noise 1.0 
        --prob_morph 0.0 
        --prob_dropout 0.0 
        --prob_false_pos 0.0
"""

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from utils.augmentations import PromptUNetAugmenter
from data.DataGenerator import DataGenerator
from data.DataLoader_npz import DataLoader_npz

def print_statistics(name, tensor, file=None):
    """Prints tensor statistics to console and optionally to a file."""
    arr = tensor.numpy()
    unique_vals = np.unique(arr)
    
    lines = [
        f"--- {name} ---",
        f"Shape: {arr.shape}",
        f"Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}, Std: {arr.std():.4f}"
    ]
    if len(unique_vals) <= 15:
        lines.append(f"Unique values: {unique_vals}")
    else:
        lines.append(f"Unique values (count): {len(unique_vals)} (Continuous data)")
    lines.append("")
    
    output_str = "\n".join(lines)
    print(output_str)
    if file:
        file.write(output_str + "\n")

def main():
    parser = argparse.ArgumentParser(description="Test PromptUNet Data Augmentations")
    parser.add_argument("--output_dir", type=str, default="utils/results", help="Directory to save plots and stats")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per modality")
    parser.add_argument("--prob", type=float, default=1.0, help="Global default probability for all augmentations")
    
    # Individual probability overrides
    parser.add_argument("--prob_photo", type=float, default=None)
    parser.add_argument("--prob_gamma", type=float, default=None)
    parser.add_argument("--prob_noise", type=float, default=None)
    parser.add_argument("--prob_independent_noise", type=float, default=None)
    parser.add_argument("--prob_morph", type=float, default=None)
    parser.add_argument("--prob_dropout", type=float, default=None)
    parser.add_argument("--prob_false_pos", type=float, default=None)
    
    args = parser.parse_args()

    # Determine final probabilities (use global --prob unless specific override is given)
    def get_prob(val):
        return val if val is not None else args.prob

    probs = {
        "prob_photo": get_prob(args.prob_photo),
        "prob_gamma": get_prob(args.prob_gamma),
        "prob_noise": get_prob(args.prob_noise),
        "prob_independent_noise": get_prob(args.prob_independent_noise),
        "prob_morph": get_prob(args.prob_morph),
        "prob_dropout": get_prob(args.prob_dropout),
        "prob_false_pos": get_prob(args.prob_false_pos)
    }

    augmenter = PromptUNetAugmenter(**probs)
    
    modalities = ["CT", "MRI"]
    npz_files = {
        "CT": "data/test_data/HanSeg_CT.npz",
        "MRI": "data/test_data/HanSeg_MRI.npz"
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for mod in modalities:
        mod_output_dir = os.path.join(args.output_dir, mod)
        os.makedirs(mod_output_dir, exist_ok=True)
        stats_file_path = os.path.join(args.output_dir, f"stats_{mod}.txt")
        
        print(f"\n{'='*20} Processing {mod} {'='*20}")
        print(f"Loading {npz_files[mod]}...")
        
        # Init dataloader
        dataloader = DataLoader_npz([npz_files[mod]], val_size=0.0, max_img=args.num_samples)
        datagenerator = DataGenerator(dataloader)
        
        ds, _ = datagenerator.get_data_points(
            max_data_points=args.num_samples, 
            offset=10, 
            max_number_labels=3, 
            cropping=False
        )

        with open(stats_file_path, "w") as f_stats:
            f_stats.write(f"Augmentation Test Statistics for {mod}\n")
            f_stats.write(f"Probabilities: {probs}\n")
            f_stats.write("="*40 + "\n\n")

            for i, (x, y, p) in enumerate(ds):
                print(f"--- [{mod}] DP {i+1} ---")
                print_statistics(f"Original x {i+1}", x, file=f_stats)
                
                x_aug, y_aug, p_aug = augmenter(x, y, p)
                
                print_statistics(f"Augmented x {i+1}", x_aug, file=f_stats)
                print_statistics(f"Augmented y {i+1}", y_aug, file=f_stats)
                
                # Prepare Plot
                p_x_orig, p_y_orig = tf.split(p, 2, axis=-1)
                p_x_aug, p_y_aug = tf.split(p_aug, 2, axis=-1)
                
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f"{mod} Augmentation Test - DP {i+1}", fontsize=16)
                
                # Plotting Row 0 (Original) and Row 1 (Augmented)
                titles = ["Original x", "Original y", "Original p_x", "Original p_y",
                          "Augmented x", "Augmented y", "Augmented p_x", "Augmented p_y"]
                images = [x, y, p_x_orig, p_y_orig, x_aug, y_aug, p_x_aug, p_y_aug]
                
                for idx, (img, title) in enumerate(zip(images, titles)):
                    ax = axes[idx // 4, idx % 4]
                    ax.imshow(np.squeeze(img), cmap='gray')
                    ax.set_title(title)
                    ax.axis('off')

                plt.tight_layout()
                save_path = os.path.join(mod_output_dir, f"aug_test_{i+1:02d}.png")
                plt.savefig(save_path)
                plt.close(fig)
                
            print(f"Finished {mod}. Visualizations saved to {mod_output_dir}")
            print(f"Statistics saved to {stats_file_path}")

    print("\nAll modalities processed.")

if __name__ == "__main__":
    main()
