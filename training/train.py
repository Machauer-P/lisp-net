"""
Training script for the Prompt U-Net model.
"""

import os
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mlflow
import tensorflow as tf

# Set path to allow importing from parent directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from data.DataLoader_npz import DataLoader_npz
from data.DataGenerator import DataGenerator

from utils.augmentations import PromptUNetAugmenter
from utils.metrics import dice_score_tf
from utils.visualization import plot_result

from prompt_unet import PromptUNet
from optimizer import PromptUNetOptimizer

# Mixed Precision setup
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- Parameters (same as p_unet_291.ipynb) ---
epochs = 4000
batch_size = 128
dp_training = 3500
dp_testing = 1000

# Additional sampling parameters
offset = 12
max_number_labels = 4

# Execution controls
new_ds = 75          # Every 'x' epochs a new dataset is generated
new_val_loop = 300   # Every 'x' epochs the validation loop is performed

version = 'p_unet_291'

# Dataset Paths
dataset_paths = [
    "data/train_data/nako_combined.npz",
    "data/train_data/total_seg_combined.npz",
    "data/train_data/msd_combined.npz"
]


def main():
    augmenter = PromptUNetAugmenter()

    # 1. Initialize DataLoader and Generator
    print("Initializing DataLoader and Generator...")
    dataloader = DataLoader_npz(dataset_paths, val_size=0.015)
    datagenerator = DataGenerator(dataloader)

    # 2. Build the Model
    print("Building model...")
    model = PromptUNet(height=datagenerator.height, width=datagenerator.width)

    # 3. Setup Optimizer with Scheduler
    print("Configuring Optimizer...")
    optimizer_config = PromptUNetOptimizer(
        epochs=epochs,
        batch_size=batch_size,
        dp_training=dp_training
    )
    model.optimizer = optimizer_config.get_optimizer()

    # Warm-up forward pass to fully initialise the network
    dummy_input = tf.random.uniform([1, datagenerator.height, datagenerator.width, 1])
    dummy_prompt = tf.random.uniform([1, datagenerator.height, datagenerator.width, 2])
    _ = model.this([dummy_input, dummy_prompt])

    # ------------------------------------------------------------------ #
    #  Persistent data buffer – populated by refresh_train_data()         #
    #  The tf.data pipeline is built ONCE and reads from this dict.       #
    #  Only the numpy arrays are swapped; no graph nodes are ever added.  #
    # ------------------------------------------------------------------ #
    _buf: dict = {'x': None, 'y': None, 'p': None, 'm': None}

    def refresh_train_data():
        """Pull a fresh random training dataset into the numpy buffer."""
        print("Pulling new random Training Dataset...")
        x_np, y_np, p_np, m_np, _ = datagenerator.get_data_points_numpy(
            max_data_points=dp_training,
            offset=offset,
            max_number_labels=max_number_labels
        )
        _buf['x'] = x_np
        _buf['y'] = y_np
        _buf['p'] = p_np
        _buf['m'] = m_np
        gc.collect()  # Free Python-side list temporaries

    def _data_gen():
        """Python generator that yields one sample at a time from _buf."""
        n = len(_buf['x'])
        indices = np.random.permutation(n)
        for i in indices:
            yield _buf['x'][i], _buf['y'][i], _buf['p'][i], _buf['m'][i]

    H, W = datagenerator.height, datagenerator.width

    # Build the pipeline graph ONCE – this is the only .map(augmenter) call
    # that will ever exist for the entire training run.
    train_ds = (
        tf.data.Dataset.from_generator(
            _data_gen,
            output_signature=(
                tf.TensorSpec(shape=(H, W, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(H, W, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(H, W, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(),        dtype=tf.float32),
            )
        )
        .map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Setup MLflow
    mlflow.set_experiment(version)
    with mlflow.start_run() as run:
        # Log basic parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_number_labels", max_number_labels)
        mlflow.log_param("num_epochs", epochs)
        mlflow.log_param("loss_function", "binary_crossentropy")
        mlflow.log_param("dp_training", dp_training)

        print("Obtaining validation datasets...")
        test_ds, _ = datagenerator.get_val_data_points(
            max_data_points=dp_testing,
            offset=offset,
            max_number_labels=max_number_labels
        )
        test_ds = test_ds.batch(1)

        # Prime the buffer before the epoch loop starts
        refresh_train_data()

        for epoch in range(epochs):

            # --- Log LR ---
            current_lr = model.optimizer.learning_rate
            if isinstance(current_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = current_lr(epoch)
            else:
                current_lr = float(current_lr.numpy())
            mlflow.log_metric("learning_rate", float(current_lr), step=epoch)

            # Reset loss metric
            model.train_loss.reset_state()

            # Save model checkpoint
            if epoch % 8 == 0 and epoch != 0:
                model_name = f'{version}.keras'
                try:
                    model_dir = "saved_models"
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, model_name)
                    model.this.save(model_path)
                except Exception as e:
                    print(f"Warning: Could not save model at epoch {epoch}. Error: {e}")

            # Validation Loop
            if epoch % new_val_loop == 0 and epoch != 0:
                total_dice = 0
                for z in test_ds:
                    val_pred = model.this([z[0], z[2]], training=False)
                    total_dice += dice_score_tf(z[1][..., 0:1], val_pred)

                avg_dice = total_dice / dp_testing
                mlflow.log_metric("validation_loss", 1 - float(avg_dice), step=epoch)
                print(f'Validation loss: {1 - float(avg_dice):.4f}')

            # Refresh data buffer every new_ds epochs.
            # The tf.data pipeline graph is NOT rebuilt – only the numpy buffer changes.
            if epoch % new_ds == 0:
                # Visualise a validation prediction and log to MLflow
                z_test = next(iter(test_ds))
                pred = model.this([z_test[0], z_test[2]], training=False)
                fig = plot_result(
                    z_test[0][0], z_test[1][0], z_test[2][0], pred[0],
                    offset, "Prediction", show=False
                )
                if fig is not None:
                    mlflow.log_figure(fig, f"predictions/epoch_{epoch}.png")
                    plt.close(fig)

                # Swap numpy buffer – no del, no pipeline rebuild needed
                refresh_train_data()

            # Train one epoch – dataset is already mapped/batched/prefetched
            model.train_epoch(train_dataset=train_ds)

            epoch_loss = float(model.train_loss.result())
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.6f}')
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)


if __name__ == "__main__":
    main()
