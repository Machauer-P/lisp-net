import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Assume this will be run from a notebook within evaluation/eval_prompt_unet
# so we append the project root directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path
from utils.Helpers import Helpers
from data.DataLoader_npz import DataLoader_npz
from data.DataGenerator import DataGenerator
from utils.augmentations import PromptUNetAugmenter
from utils.visualization import plot_result, visualize_a_few_results

# Calculate project root (assuming script is in evaluation/eval_prompt_unet/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class PromptUNetTester:
    def __init__(self, dataset_path, models_dir, augmentations=None, max_data_points=1000):
        # Resolve dataset_path relative to root if needed
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
        
        self.dataset_path = [str((PROJECT_ROOT / p).resolve()) if not os.path.isabs(p) else p for p in dataset_path]
            
        # Resolve models_dir relative to root if needed
        if not os.path.isabs(models_dir):
            self.models_dir = str((PROJECT_ROOT / models_dir).resolve())
        else:
            self.models_dir = models_dir
        self.augmentations = augmentations if augmentations is not None else []
        self.max_data_points = max_data_points
        self.helpers = Helpers()


    def test_routine(self, model_name: str, loaded_model: tf.keras.Model, ds, offset, threshold=0.45):
        start = time.time()
        total_dice = 0
        len_ds = len(list(ds.as_numpy_iterator()))

        for e, (x, y, p) in enumerate(ds):
            
            x = self.helpers.shaping(x)
            p = self.helpers.shaping(p)
                
            pred = loaded_model.predict([x[0:1, :, :, 0:1], p[0:1, ...]], verbose=0)

            pred = tf.where(pred < threshold, 0.0, pred)
            pred = tf.where(pred >= threshold, 1.0, pred)
            current_dice = self.helpers.dice_score_tf(y[..., 0:1], pred)
            total_dice += current_dice

        end = time.time()
        print()
        print(f'Testing for {model_name} took {round(end - start, 0)} seconds')

        return (total_dice / len_ds).numpy()

    def run_pipeline(self, dimensions, offsets, models, threshold=0.45, max_number_labels=10, cropping=False, min_crop_size=0.5, cropping_composition=1, num_visualize=0):
        results = {}
        
        # Initialize DataLoader_npz and DataGenerator
        dataloader = DataLoader_npz(self.dataset_path, val_size=0.0)
        datagenerator = DataGenerator(dataloader)
        
        for off in offsets:
            print(f"\n" + "="*50)
            print(f"   EVALUATING OFFSET: {off} | AXIS: {dimensions}")
            print("="*50 + "\n")
            
            # 1. Generate Dataset for this offset
            test_ds, offset_list = datagenerator.get_data_points(
                max_data_points=self.max_data_points, 
                offset=off, 
                max_number_labels=max_number_labels, 
                dimensions=dimensions, 
                cropping=cropping,
                min_crop_size=min_crop_size, 
                cropping_composition=cropping_composition
            )
            
            if self.augmentations:
                augmenter = PromptUNetAugmenter(test_ds, self.augmentations)
                test_ds = augmenter.process()
            
            # Cache for visualizations to avoid reloading models twice
            model_cache = []

            # 2. Section: Performance Metrics
            print()
            print(f"--- [SECTION 1: METRICS] ---")
            for model_name in models:
                try:
                    model_path = os.path.join(self.models_dir, model_name)
                    loaded_model = tf.keras.models.load_model(model_path, compile=False)
                    
                    avg_dice = self.test_routine(
                        model_name=model_name, 
                        loaded_model=loaded_model, 
                        ds=test_ds, 
                        offset=off, 
                        threshold=threshold
                    )
                    
                    print(f"[{model_name}] -> Avg Dice: {avg_dice:.3f}")
                    
                    if model_name not in results:
                        results[model_name] = []
                    
                    results[model_name].append({
                        "offset": off,
                        "axis": dimensions,
                        "avg_dice": avg_dice
                    })

                    # Store for visualization phase
                    if num_visualize > 0:
                        model_cache.append((model_name, loaded_model))
                    
                except Exception as e:
                    print(f"Error testing model {model_name} on offset {off}: {e}")
            
            # 3. Section: Visualizations (if requested)
            if num_visualize > 0 and model_cache:
                print(f"\n--- [SECTION 2: VISUALIZATIONS] ---")
                for model_name, loaded_model in model_cache:
                    print(f"\n>> Visualizing predictions for: {model_name} (Threshold: {threshold})")
                    visualize_a_few_results(
                        model_name=model_name,
                        loaded_model=loaded_model,
                        ds=test_ds,
                        offset=offset_list,
                        img_to_plot=num_visualize,
                        threshold=threshold
                    )
                    
        return results
