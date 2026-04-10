import os
import sys
import pickle
import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.test_data.ds_handler_2d import load_tf_dataset_2D
from utils.metrics import dice_numpy, dice_score_tf
from utils.preprocessing import shaping
from evaluation.benchmark_models.UniverSeg.universeg import universeg as load_universeg_model

class EvalPipeline2D:
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Configure TF GPU growth if on native Windows
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def load_universeg(self, pretrained=True):
        """Loads and prepares UniverSeg model."""
        model = load_universeg_model(pretrained=pretrained)
        model.to(self.device)
        model.eval()
        return model

    def load_prompt_unet(self, model_version='21'):
        """Loads a specific version of Prompt-UNet."""
        model_path = PROJECT_ROOT / f'training/p_unet_{model_version}.keras'
        if not model_path.exists():
            # Alternative path check
            model_path = PROJECT_ROOT / f'prompt_unet/prompt_unet{model_version}.keras'
            
        if not model_path.exists():
            raise FileNotFoundError(f"Prompt U-Net model not found at {model_path}")
            
        return tf.keras.models.load_model(str(model_path))

    def discover_test_sets(self, data_path):
        """
        Scans data_path for pairs of TFRecords.
        Pairs are identified by [num]_[name]_support.tfrecord and [num]_[name].tfrecord.
        """
        data_path = Path(data_path)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
            
        support_files = list(data_path.glob("*_support.tfrecord"))
        pairs = []
        
        for s_file in support_files:
            # Strip _support.tfrecord suffix
            basename = s_file.name.replace("_support.tfrecord", "")
            query_file = data_path / f"{basename}.tfrecord"
            
            if query_file.exists():
                pairs.append({
                    'index': basename.split('_')[0],
                    'name': basename,
                    'query': query_file,
                    'support': s_file
                })
        
        # Sort by index if possible
        try:
            pairs.sort(key=lambda x: int(x['index']))
        except ValueError:
            pairs.sort(key=lambda x: x['index'])
            
        return pairs

    def load_support_set(self, support_file_path):
        """Loads support images and labels from a TFRecord for UniverSeg."""
        path = Path(support_file_path).parent
        filename = Path(support_file_path).stem # remove .tfrecord
        
        # load_tf_dataset_2D handles extension if missing
        ds_support = load_tf_dataset_2D(filename, str(path), include_prompt=False, include_offset=False)
        
        images = []
        labels = []
        
        # UniverSeg expects 16 support points
        for i, (x, y) in enumerate(ds_support):
            if i >= 16:
                break
            
            # Ensure (128, 128)
            x_np = x.numpy()
            y_np = y.numpy()
            if x_np.ndim == 3: x_np = x_np.squeeze(-1)
            if y_np.ndim == 3: y_np = y_np.squeeze(-1)
            
            images.append(x_np)
            labels.append(y_np)
            
        if len(images) < 16:
            print(f"Warning: Support set only contains {len(images)} points.")

        # Prepare for UniverSeg: (16, 1, 128, 128)
        support_images = torch.from_numpy(np.stack(images, axis=0)).float().unsqueeze(1).to(self.device)
        support_labels = torch.from_numpy(np.stack(labels, axis=0)).float().unsqueeze(1).to(self.device)
        
        return support_images, support_labels

    def evaluate_pair(self, pair, u_seg_model, p_unet_model, threshold=0.45):
        """Evaluates a single pair of query and support datasets."""
        query_path = pair['query'].parent
        query_name = pair['query'].stem
        
        # Load query dataset with offsets inside
        test_ds = load_tf_dataset_2D(query_name, str(query_path), include_offset=True, include_prompt=True)
        
        # Load support set for UniverSeg
        support_images, support_labels = self.load_support_set(pair['support'])
        
        prompt_unet_dices = []
        universeg_dices = []
        
        for x, y, p, offset in test_ds:
            # --- Prompt-UNet Inference ---
            # shaping adds batch and channel dims if needed: (1, 128, 128, 1)
            x_p = shaping(x)
            p_p = shaping(p)
            
            p_unet_pred = p_unet_model.predict([x_p, p_p], verbose=0)
            p_unet_pred = (p_unet_pred >= threshold).astype(np.float32)
            
            dice_p = dice_numpy(y, p_unet_pred)
            prompt_unet_dices.append(dice_p)
            
            # --- UniverSeg Inference ---
            # Prepare x_u: Ensure (1, 1, 128, 128)
            x_u = torch.from_numpy(x.numpy()).float()
            if x_u.dim() == 2:  # (H, W)
                x_u = x_u.unsqueeze(0).unsqueeze(0)
            elif x_u.dim() == 3: # (H, W, C)
                x_u = x_u.permute(2, 0, 1).unsqueeze(0)
            x_u = x_u.to(self.device)
            
            with torch.no_grad():
                logits = u_seg_model(x_u, support_images[None], support_labels[None])[0]
                u_pred = torch.sigmoid(logits)
                u_pred = (u_pred > threshold).float()
            
            dice_u = dice_numpy(y, u_pred)
            universeg_dices.append(dice_u)
            
        return np.mean(prompt_unet_dices), np.mean(universeg_dices)

    def run_full_evaluation(self, data_path, p_unet_version='21', output_file=None):
        """Runs the complete comparison pipeline."""
        print(f"Starting evaluation on: {data_path}")
        pairs = self.discover_test_sets(data_path)
        print(f"Discovered {len(pairs)} dataset pairs.")
        
        print("Loading models...")
        u_seg = self.load_universeg()
        p_unet = self.load_prompt_unet(p_unet_version)
        
        results = []
        
        for pair in tqdm(pairs, desc="Evaluating datasets"):
            dice_p, dice_u = self.evaluate_pair(pair, u_seg, p_unet)
            results.append({
                'index': pair['index'],
                'name': pair['name'],
                'prompt_unet_dice': dice_p,
                'universeg_dice': dice_u
            })
            
        avg_p = np.mean([r['prompt_unet_dice'] for r in results])
        avg_u = np.mean([r['universeg_dice'] for r in results])
        
        print(f"\nFinal Results:")
        print(f"Prompt U-Net (V{p_unet_version}) Mean Dice: {avg_p:.4f}")
        print(f"UniverSeg Mean Dice: {avg_u:.4f}")
        
        if output_file:
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to {output_file}")
            
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/test_data/2d/offset_5")
    parser.add_argument("--p_unet_version", type=str, default="21")
    parser.add_argument("--output", type=str, default="evaluation/benchmark_universeg/eval_results.pkl")
    args = parser.parse_args()
    
    pipeline = EvalPipeline()
    pipeline.run_full_evaluation(args.data_path, args.p_unet_version, args.output)
