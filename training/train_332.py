"""
Prompt U-Net v332 — Final Model — Standalone Training Script
=============================================================

Usage (from project root or training/ directory):
    python training/train_332.py

All hyper-parameters are defined in the CONFIGURATION block below.
MLflow metrics are written to the same mlruns/ database as the notebooks.
"""

import os
import sys
import gc
import logging
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # training/
_PROJECT_ROOT = _HERE.parent                       # prompt-unet/
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import mlflow
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Project imports ──────────────────────────────────────────────────────────
from data.DataLoader_npz import DataLoader_npz
from data.DataGenerator  import DataGenerator

from utils.augmentations import PromptUNetAugmenter
from utils.metrics       import dice_score_tf
# from utils.visualization import plot_result

from training.prompt_unet_313 import PromptUNet          # v313 architecture
from training.optimizer        import PromptUNetOptimizer  # WarmupFlatCosineDecay

# ============================================================================
# CONFIGURATION — edit only this block
# ============================================================================

VERSION           = "p_unet_332"

EPOCHS            = 4000
BATCH_SIZE        = 128
DP_TRAINING       = 10_000    # 10 k points per buffer refresh
DP_TESTING        = 1_000

OFFSET            = 16         # slice-distance offset (v316)
MAX_NUMBER_LABELS = 4

NEW_DS            = 30         # refresh training data every N epochs
NEW_VAL_LOOP      = 300        # run validation every N epochs

WARMUP_EPOCHS     = 50
FLAT_EPOCHS       = 1_500

DATASET_PATHS = [
    "data/train_data/nako_combined.npz",       # 61 PIDs
    "data/train_data/total_seg_combined.npz",  # 45 PIDs
    "data/train_data/msd_combined.npz",        # 40 PIDs
    "data/train_data/brats_gli.npz",           # 20 PIDs
    "data/train_data/brats_men_rt.npz",        #  6 PIDs
    "data/train_data/TopCoW_MR.npz",           # 18 PIDs  (from v319)
    "data/train_data/TopCoW_CT.npz",           # 18 PIDs  (from v319)
]  # Total: 208 patients

# ============================================================================
# END CONFIGURATION
# ============================================================================


def main():
    print(f"TF  : {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # Resolve dataset paths relative to project root so the script can be
    # invoked from any working directory.
    abs_paths = [str(_PROJECT_ROOT / p) for p in DATASET_PATHS]
    dataloader    = DataLoader_npz(abs_paths, val_size=0.01)
    datagenerator = DataGenerator(dataloader)
    H, W = datagenerator.height, datagenerator.width
    print(f"Image size: {H} x {W}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PromptUNet(height=H, width=W)
    # Loss stays as default binary_crossentropy (set inside PromptUNet.__init__)

    # Warm-up forward pass to fully initialise all layers
    _ = model.this([
        tf.random.uniform([1, H, W, 1]),
        tf.random.uniform([1, H, W, 2]),
    ])
    print(f"Trainable params: {model.this.count_params():,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt_builder = PromptUNetOptimizer(
        epochs        = EPOCHS,
        batch_size    = BATCH_SIZE,
        dp_training   = DP_TRAINING,
        warmup_epochs = WARMUP_EPOCHS,
        flat_epochs   = FLAT_EPOCHS,
    )
    model.optimizer = opt_builder.get_optimizer()
    steps_per_epoch = opt_builder.steps_per_epoch

    # ── Augmentation ──────────────────────────────────────────────────────────
    augmenter = PromptUNetAugmenter(
        prob_photo             = 0.45,
        prob_gamma             = 0.35,
        prob_noise             = 0.40,
        prob_independent_noise = 0.50,
        prob_geometric         = 0.50,
        prob_morph             = 0.30,
        prob_dropout           = 0.40,
        prob_false_pos         = 0.60,
        gamma_range                 = (0.85, 1.25),
        noise_std_range             = (0.0, 0.10),
        independent_noise_std_range = (0.0, 0.01),
    )

    # ── Persistent tf.data pipeline ───────────────────────────────────────────
    _buf: dict = {"x": None, "y": None, "p": None, "m": None}

    def refresh_train_data():
        x_np, y_np, p_np, m_np, _ = datagenerator.get_data_points_numpy(
            max_data_points   = DP_TRAINING,
            offset            = OFFSET,
            max_number_labels = MAX_NUMBER_LABELS,
        )
        _buf["x"] = x_np
        _buf["y"] = y_np
        _buf["p"] = p_np
        _buf["m"] = m_np
        gc.collect()

    def _data_gen():
        n       = len(_buf["x"])
        indices = np.random.permutation(n)
        for i in indices:
            yield _buf["x"][i], _buf["y"][i], _buf["p"][i], _buf["m"][i]

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
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("Pipeline ready.")

    # ── Training loop ─────────────────────────────────────────────────────────
    # Change working directory to training/ so that MLflow writes mlruns/ and
    # Keras checkpoints into the same directory as the notebooks.
    os.chdir(_HERE)

    mlflow.set_experiment(VERSION)

    with mlflow.start_run():

        mlflow.log_params({
            "batch_size"        : BATCH_SIZE,
            "max_number_labels" : MAX_NUMBER_LABELS,
            "num_epochs"        : EPOCHS,
            "dp_training"       : DP_TRAINING,
            "offset"            : OFFSET,
            "loss_function"     : "binary_crossentropy",
            "new_ds"            : NEW_DS,
            "warmup_epochs"     : WARMUP_EPOCHS,
            "flat_epochs"       : FLAT_EPOCHS,
            "prob_geometric"    : augmenter.prob_geometric,
            "prob_morph"        : augmenter.prob_morph,
            "gamma_range"       : str(augmenter.gamma_range),
            "trainable_params"  : model.this.count_params(),
            "scale_augmentation": "50% crop 128px / 50% crop [128,256]px resized",
            "se_attention"      : "enabled",
            "mixed_precision"   : "false",
            "datasets"          : "nako+total_seg+msd+brats_gli+brats_men_rt+TopCoW_MR+TopCoW_CT",
        })

        # Validation dataset (built once, no augmentation)
        val_x, val_y, val_p, val_m, _ = datagenerator.get_val_data_points_numpy(
            max_data_points   = DP_TESTING,
            offset            = OFFSET,
            max_number_labels = MAX_NUMBER_LABELS,
        )
        test_ds = (
            tf.data.Dataset.from_tensor_slices((val_x, val_y, val_p, val_m))
            .batch(1)
        )

        # Prime the training buffer
        refresh_train_data()

        for epoch in range(EPOCHS):

            model.train_loss.reset_state()

            # Log learning rate
            lr = model.optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = float(lr(epoch * steps_per_epoch))
            else:
                lr = float(lr.numpy())
            mlflow.log_metric("learning_rate", lr, step=epoch)

            # Checkpoint every 8 epochs
            if epoch % 8 == 0 and epoch != 0:
                model.this.save(f"{VERSION}.keras")

            # Validation every NEW_VAL_LOOP epochs
            if epoch % NEW_VAL_LOOP == 0 and epoch != 0:
                total_dice = 0.0
                for z in test_ds:
                    pred = model.this([z[0], z[2]], training=False)
                    total_dice += float(dice_score_tf(z[1][..., 0:1], pred))
                val_loss = 1.0 - total_dice / DP_TESTING
                mlflow.log_metric("validation_loss", val_loss, step=epoch)
                print(f"  Validation loss: {val_loss:.4f}")

            # Refresh training data every NEW_DS epochs
            if epoch % NEW_DS == 0 and epoch != 0:
                refresh_train_data()

            # Train one epoch
            model.train_epoch(train_dataset=train_ds)

            epoch_loss = float(model.train_loss.result())
            print(f"Epoch {epoch + 1:>4d}  loss: {epoch_loss:.6f}")
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

    # Final checkpoint
    model.this.save(f"{VERSION}.keras")
    print(f"\nTraining complete. Model saved as {VERSION}.keras")


if __name__ == "__main__":
    main()
