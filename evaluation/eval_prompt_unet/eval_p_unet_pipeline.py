import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import sys
import time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib.pyplot as plt
import re

# Assume this will be run from a notebook within evaluation/eval_prompt_unet
# so we append the project root directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path
from data.DataLoader_npz import DataLoader_npz
from data.DataGenerator import DataGenerator


from utils.visualization import visualize_a_few_results

# Calculate project root (assuming script is in evaluation/eval_prompt_unet/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class PromptUNetTester:
    def __init__(self, dataset_path, models_dir, max_data_points=1000):
        # Resolve dataset_path relative to root if needed
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
        
        self.dataset_path = [str((PROJECT_ROOT / p).resolve()) if not os.path.isabs(p) else p for p in dataset_path]
            
        # Resolve models_dir relative to root if needed
        if not os.path.isabs(models_dir):
            self.models_dir = str((PROJECT_ROOT / models_dir).resolve())
        else:
            self.models_dir = models_dir
        self.max_data_points = max_data_points


    def test_routine(self, model_name: str, loaded_model: tf.keras.Model, ds, offset, threshold=0.45):
        total_dice = 0.0
        count = 0

        is_old_model = False
        m_ver = re.search(r'p_unet_(\d+)', model_name)
        if m_ver and int(m_ver.group(1)) < 292:
            is_old_model = True

        threshold_t = tf.constant(threshold, dtype=tf.float32)

        # Build a fused @tf.function: preprocessing + inference + Dice in one graph.
        # Call loaded_model() directly (not via a nested tf.function like predictor._fast_predict_fn)
        # so TF can fully trace and inline all model ops. A nested tf.function call creates
        # an opaque function-call op in the outer graph that prevents inlining.
        if is_old_model:
            # Vectorized robust min-max norm across a batch.
            # Replaces tf.map_fn(min_max_norm, ...) which ran tfp.stats.percentile
            # serially per image (64 calls/batch). This sorts the full batch tensor
            # once and indexes the percentile positions — fully vectorized.
            def _batch_min_max_norm(t):
                shape = tf.shape(t)
                flat = tf.reshape(t, [shape[0], -1])
                flat_sorted = tf.sort(flat, axis=1)
                N = tf.cast(tf.shape(flat_sorted)[1], tf.float32)
                lo_idx = tf.cast(tf.round(0.005 * (N - 1)), tf.int32)
                hi_idx = tf.cast(tf.round(0.995 * (N - 1)), tf.int32)
                q_min = tf.reshape(flat_sorted[:, lo_idx:lo_idx+1], [shape[0], 1, 1, 1])
                q_max = tf.reshape(flat_sorted[:, hi_idx:hi_idx+1], [shape[0], 1, 1, 1])
                return (tf.clip_by_value(t, q_min, q_max) - q_min) / (q_max - q_min + 1e-8)

            @tf.function
            def eval_step(x, y, p):
                x_norm     = _batch_min_max_norm(x)
                p_img_norm = _batch_min_max_norm(p[..., 0:1])
                p_in       = tf.concat([p_img_norm, p[..., 1:2]], axis=-1)
                pred       = tf.cast(loaded_model([x_norm[..., 0:1], p_in], training=False), tf.float32)
                pred_bin   = tf.cast(pred >= threshold_t, tf.float32)
                true_mask  = tf.cast(y[..., 0:1], tf.float32)
                axes       = [1, 2, 3]
                intersection = tf.reduce_sum(true_mask * pred_bin, axis=axes)
                denominator  = tf.reduce_sum(true_mask, axis=axes) + tf.reduce_sum(pred_bin, axis=axes)
                dice_scores  = (2. * intersection + 1e-6) / (denominator + 1e-6)
                return tf.reduce_sum(dice_scores), tf.shape(x)[0]
        else:
            @tf.function
            def eval_step(x, y, p):
                pred      = tf.cast(loaded_model([x[..., 0:1], p], training=False), tf.float32)
                pred_bin  = tf.cast(pred >= threshold_t, tf.float32)
                true_mask = tf.cast(y[..., 0:1], tf.float32)
                axes      = [1, 2, 3]
                intersection = tf.reduce_sum(true_mask * pred_bin, axis=axes)
                denominator  = tf.reduce_sum(true_mask, axis=axes) + tf.reduce_sum(pred_bin, axis=axes)
                dice_scores  = (2. * intersection + 1e-6) / (denominator + 1e-6)
                return tf.reduce_sum(dice_scores), tf.shape(x)[0]

        # Batch the dataset to eliminate per-item kernel launch overhead.
        batched_ds = ds.batch(64)

        # Warmup — trace eval_step with the actual batch shape used in the hot loop.
        warmup_item = next(iter(batched_ds))
        x_w, y_w, p_w = warmup_item[0], warmup_item[1], warmup_item[2]
        eval_step(x_w, y_w, p_w)

        print(f"Testing {model_name}... ", end="", flush=True)
        start = time.time()

        for item in batched_ds:
            if len(item) == 4:
                x, y, p, m = item
            else:
                x, y, p = item

            dice_sum, n = eval_step(x, y, p)
            total_dice += dice_sum.numpy()
            count      += n.numpy()

        end = time.time()
        avg_dice = total_dice / count if count > 0 else 0.0

        print(f"Done. Took {round(end - start, 1)} seconds.")
        return avg_dice



    def run_pipeline(self, dimensions, offsets, models, threshold=0.45, max_number_labels=10, num_visualize=0):
        results = {}

        # On CPU, float16 has NO hardware acceleration (no Tensor Cores).
        # Mixed-precision models like v292 run slower in float16 on CPU than float32,
        # because TF must upcast float16 ops internally. Force float32 globally so
        # all models — regardless of their saved compute dtype — run at native speed.
        on_cpu = len(tf.config.list_physical_devices('GPU')) == 0
        if on_cpu:
            tf.keras.mixed_precision.set_global_policy('float32')
            print("[INFO] CPU detected — forcing float32 compute policy for all models.")

        # Initialize DataLoader_npz and DataGenerator
        dataloader = DataLoader_npz(self.dataset_path, val_size=0.0)
        datagenerator = DataGenerator(dataloader)

        for i, off in enumerate(offsets):
            print(f"\n" + "="*50)
            print(f"   EVALUATING OFFSET: {off} | AXIS: {dimensions}")
            print("="*50 + "\n")

            # 1. Generate Dataset for this offset
            test_ds, offset_list = datagenerator.get_data_points(
                max_data_points=self.max_data_points,
                offset=off,
                max_number_labels=max_number_labels,
                dimensions=dimensions,
            )

            is_last_offset = (i == len(offsets) - 1)

            # 2. Section: Performance Metrics
            # Each model is loaded, tested, then IMMEDIATELY freed from GPU memory.
            # Without this, every model accumulates in VRAM simultaneously, causing
            # the GPU allocator to fragment and thrash — which is why inference time
            # grew monotonically with each successive model.
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

                except Exception as e:
                    print(f"Error testing model {model_name} on offset {off}: {e}")

                finally:
                    # Explicitly release GPU memory before loading the next model.
                    # del + gc.collect() drops Python references; clear_session() tells
                    # the TF/Keras allocator to return the memory to the CUDA pool.
                    try:
                        del loaded_model
                    except NameError:
                        pass
                    gc.collect()
                    tf.keras.backend.clear_session()

            # 3. Section: Visualizations (if requested, reload models in a fresh pass)
            # Models are reloaded here so the visualization phase never overlaps with
            # the metrics phase, keeping GPU memory usage low throughout.
            if num_visualize > 0 and is_last_offset:
                print(f"\n--- [SECTION 2: VISUALIZATIONS] ---")
                for model_name in models:
                    try:
                        model_path = os.path.join(self.models_dir, model_name)
                        loaded_model = tf.keras.models.load_model(model_path, compile=False)

                        print(f"\n>> Visualizing predictions for: {model_name} (Threshold: {threshold})")
                        visualize_a_few_results(
                            model_name=model_name,
                            loaded_model=loaded_model,
                            ds=test_ds,
                            offset=offset_list,
                            img_to_plot=num_visualize,
                            threshold=threshold
                        )

                    except Exception as e:
                        print(f"Error visualizing model {model_name}: {e}")

                    finally:
                        try:
                            del loaded_model
                        except NameError:
                            pass
                        gc.collect()
                        tf.keras.backend.clear_session()

        return results

