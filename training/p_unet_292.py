#!/usr/bin/env python
# coding: utf-8

# # Prompt U-Net Version 292 Training

# # Changes
# 
# ## 1. Preprocessing
# 
# - **Isotropic resampling before normalization**
#   - Apply once to volumes before data generation.
# 
# - **Data loader**
#   - The training data loaders have been completely rewritten for .npz usage.
# 
# - **Volume standardization**
#   - Switch to **z-score normalization**
#   - Apply on the full volume.
# 
# - **Important design choice**
#   - CT and MRI use different normalization pipelines.
#   - CT or MRT is stored in `.npz` files.
# 
# ---
# 
# ## 2. CT Preprocessing
# 
# 1. **Intensity clipping**
#    - Range: `[-1000, 1000]`
#    - Removes extreme artifacts (e.g., metal).
# 
# 2. **Global statistics (hardcoded)**
#    - Mean: `-15`
#    - Std: `160`
# 
# 3. **Normalization**
#    - Apply z-score normalization.
# 
# ---
# 
# ## 3. MRI Preprocessing
# 
# *(MRI intensities have no physical unit)*
# 
# 1. **Percentile clipping**
#    - Applied on foreground only (`> 0`)
#    - Range: 0.5% – 99.5%
# 
# 2. **Statistics computation**
#    - Mean and std computed from the volume (foreground only)
# 
# 3. **Normalization**
#    - Apply z-score normalization.
# 
# ---
# 
# ## 4. Data Generator
# 
# - Normalization must be applied **inside datagen**
#   - Required because original (unnormalized) volumes are needed for `nnInteractive`
# 
# - Issue:
#   - Resampling inside datagen breaks isotropy again
#   - `_extract_patch_2d()` always extracts fixed `128 × 128` patches
# 
# - Fixes / changes:
#   - Cache normalization results
#   - Remove volume cropping completely
#   - Ignore samples where any axis is below a threshold
#     - Prevents distorted ("squashed") images (e.g. `192 × 192 × 10`)
#   - If a dimension `< 128`, apply padding with value **5 instead of 0**
#     - Reason: `0` would represent tissue after normalization
# 
# - Data format per sample:
#   - `(x[num_dp,128,128,1], y[num_dp,128,128,1], p[num_dp,128,128,2], m)`
#   - where `m = modality`
# 
# ---
# 
# ## 5. Performance Issues
# 
# - Datagen caused **OOM (out-of-memory) errors during training**
#   - Entire datagen pipeline switched to **NumPy implementation**
# 
# ---
# 
# ## 6. Data Augmentation
# 
# - MRI:
#   - Reduced noise strength
# 
# - CT:
#   - No noise augmentation
#   - No gamma augmentation
# 
# - General changes:
#   - Stronger random brightness augmentation
#   - Geometric augmentation probability reduced to **85%**
#     - Previously: 100%
# 
# ---
# 
# ## 7. Codebase Restructuring
# 
# - Model, optimizer, and augmentation moved to `.py` files
# - `train.py` added
# - Notebook (`.ipynb`) structure temporarily retained

# ## Setup

# In[2]:


import os
import sys
import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")

import logging
tf.get_logger().setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print(f"TF  : {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")


# In[ ]:


# Allow importing from project root
notebook_dir = Path().resolve()
project_root  = notebook_dir.parent
sys.path.insert(0, str(project_root))

from data.DataLoader_npz import DataLoader_npz
from data.DataGenerator  import DataGenerator

from utils.augmentations  import PromptUNetAugmenter
from utils.metrics        import dice_score_tf
from utils.visualization  import plot_result

from prompt_unet_292 import PromptUNet
from optimizer   import PromptUNetOptimizer


# ## Data Loading

# In[4]:


dataset_paths = [
    "data/train_data/nako_combined.npz",
    "data/train_data/total_seg_combined.npz",
    "data/train_data/msd_combined.npz",
]

dataloader    = DataLoader_npz(dataset_paths, val_size=0.01)
datagenerator = DataGenerator(dataloader)

print(f"Image size: {datagenerator.height} x {datagenerator.width}")


# ## Hyperparameters

# In[5]:


version           = "p_unet_292"

epochs            = 4000
batch_size        = 128
dp_training       = 3500
dp_testing        = 1000

offset            = 12
max_number_labels = 4

new_ds       = 75    # refresh training data every N epochs
new_val_loop = 300   # run validation every N epochs


# ## Model & Optimizer

# In[6]:


opt_config = PromptUNetOptimizer(
    epochs               = epochs,
    batch_size           = batch_size,
    dp_training          = dp_training,
    warmup_epochs         = 50,
    initial_learning_rate = 1e-6,
    warmup_target         = 1e-3,
    alpha                 = 0.01,
)

model = PromptUNet(height=datagenerator.height, width=datagenerator.width)
model.optimizer = opt_config.get_optimizer()

# Warm-up forward pass to fully initialise all layers
_dummy_x = tf.random.uniform([1, datagenerator.height, datagenerator.width, 1])
_dummy_p = tf.random.uniform([1, datagenerator.height, datagenerator.width, 2])
_ = model.this([_dummy_x, _dummy_p])

print(f"Trainable params: {model.this.count_params():,}")


# ## Augmentation Pipeline

# In[7]:


augmenter = PromptUNetAugmenter(
    prob_photo             = 0.45,
    prob_gamma             = 0.45,
    prob_noise             = 0.40,
    prob_independent_noise = 0.50,
    prob_geometric         = 0.85,
    prob_morph             = 0.60,
    prob_dropout           = 0.40,
    prob_false_pos         = 0.60,
    noise_std_range             = (0.0, 0.10),
    independent_noise_std_range = (0.0, 0.01),
)


# ## Persistent tf.data Pipeline
# 
# The pipeline graph (including `.map(augmenter)`) is built **once** here.
# When fresh training data is needed, only the numpy buffer is swapped — no TF graph nodes accumulate over time, eliminating the OOM risk.

# In[8]:


# ── Shared numpy buffer ───────────────────────────────────────────────────
_buf = {"x": None, "y": None, "p": None, "m": None}

def refresh_train_data():
    """Pull fresh random training data into the numpy buffer."""
    x_np, y_np, p_np, m_np, _ = datagenerator.get_data_points_numpy(
        max_data_points   = dp_training,
        offset            = offset,
        max_number_labels = max_number_labels,
    )
    _buf["x"] = x_np
    _buf["y"] = y_np
    _buf["p"] = p_np
    _buf["m"] = m_np
    gc.collect()


def _data_gen():
    """Yields one shuffled sample at a time from the numpy buffer."""
    n       = len(_buf["x"])
    indices = np.random.permutation(n)
    for i in indices:
        yield _buf["x"][i], _buf["y"][i], _buf["p"][i], _buf["m"][i]


H, W = datagenerator.height, datagenerator.width

# Build the pipeline graph ONCE for the entire training run
train_ds = (
    tf.data.Dataset.from_generator(
        _data_gen,
        output_signature=(
            tf.TensorSpec(shape=(H, W, 1), dtype=tf.float32),  # image
            tf.TensorSpec(shape=(H, W, 1), dtype=tf.float32),  # label
            tf.TensorSpec(shape=(H, W, 2), dtype=tf.float32),  # prompt
            tf.TensorSpec(shape=(),        dtype=tf.float32),  # modality (0=CT 1=MRI)
        )
    )
    .map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

print("Pipeline ready.")


# ## Training

# In[ ]:


def fit(epochs):
    mlflow.set_experiment(version)

    with mlflow.start_run():

        mlflow.log_params({
            "batch_size"        : batch_size,
            "max_number_labels" : max_number_labels,
            "num_epochs"        : epochs,
            "dp_training"       : dp_training,
            "offset"            : offset,
            "loss_function"     : "binary_crossentropy",
        })

        # ── Validation dataset (built once, no augmentation) ───────────────
        val_x, val_y, val_p, val_m, _ = datagenerator.get_val_data_points_numpy(
            max_data_points   = dp_testing,
            offset            = offset,
            max_number_labels = max_number_labels,
        )
        test_ds = (
            tf.data.Dataset.from_tensor_slices((val_x, val_y, val_p, val_m))
            .batch(1)
        )

        # ── Prime the training buffer before the loop ─────────────────────
        refresh_train_data()

        for epoch in range(epochs):

            model.train_loss.reset_state()

            # Log learning rate
            lr = model.optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = float(lr(epoch))
            else:
                lr = float(lr.numpy())
            mlflow.log_metric("learning_rate", lr, step=epoch)

            # Checkpoint every 8 epochs
            if epoch % 8 == 0 and epoch != 0:
                model.this.save(f"{version}.keras")

            # Validation every new_val_loop epochs
            if epoch % new_val_loop == 0 and epoch != 0:
                total_dice = 0.0
                for z in test_ds:
                    pred = model.this([z[0], z[2]], training=False)
                    total_dice += float(dice_score_tf(z[1][..., 0:1], pred))
                val_loss = 1.0 - total_dice / dp_testing
                mlflow.log_metric("validation_loss", val_loss, step=epoch)
                print(f"  Validation loss: {val_loss:.4f}")

            # Refresh training data every new_ds epochs
            if epoch % new_ds == 0 and epoch != 0:
                # Visualise one validation prediction
                z_test = next(iter(test_ds))
                pred   = model.this([z_test[0], z_test[2]], training=False)
                plot_result(z_test[0][0], z_test[1][0], z_test[2][0], pred[0], offset, "")

                # Swap numpy buffer — pipeline graph stays intact
                refresh_train_data()

            # Train one epoch (dataset is already mapped/batched/prefetched)
            model.train_epoch(train_dataset=train_ds)

            epoch_loss = float(model.train_loss.result())
            print(f"Epoch {epoch + 1:>4d}  loss: {epoch_loss:.6f}")
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)


fit(epochs)


# In[ ]:




