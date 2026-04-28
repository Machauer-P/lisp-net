# Model Development Changelog

This document tracks the evolution of the Prompt U-Net segmentation model, from architectural changes to preprocessing and data augmentation strategies.

## [v330] - nnUNet-Inspired Loss & Scaled Data
*Implementation: `loss.py`, `p_unet_330.ipynb`*

**Objective Function Overhaul**
- **Batch Dice + BCE Loss:** Replaced standard Binary Cross-Entropy with a combined `DiceBCELoss`. This implements a "Batch Soft Dice" which computes the Dice coefficient globally across the entire flattened batch. This provides much more stable gradients than per-sample Dice.

**Training Configuration Scaling**
- **Increased Training Buffer:** Scaled the data generator from 3,500 to 10,000 data points per buffer refresh to improve generalization and expose the model to more unique anatomical patches.
- **Increased Refresh Frequency:** Increased the refresh rate (`new_ds`) from every 50 epochs to every 20 epochs to reduce the "sawtooth" overfitting pattern observed in validation metrics.
- **Base Model:** Inherited the architecture and pipeline from v315 (Float32 + SE Attention + BraTS datasets), as it was identified as the best performing stable baseline.

## [v316] - Large Offset Variant
*Implementation: `p_unet_316.ipynb`*

**Hyperparameters**
- **Increased Offset:** The slice distance offset was increased from 12 (v315) to 16.
- **Identical to v315 otherwise:** Uses the same BraTS-augmented data pipeline and the v313 architecture (Float32 + SE Attention).

## [v315] - Addition of BraTS Datasets
*Implementation: `prompt_unet_313.py`, `p_unet_315.ipynb`*

**Data Additions**
- **BraTS Inclusion:** Added subsets of BraTS datasets to the training pipeline to increase anatomical variety and introduce brain tumor scenarios.
  - Extracted 20 random patients across the 4 modalities of `BraTS_GLI` (`t1c`, `t1n`, `t2f`, `t2w`) into `brats_gli_training.npz`.
  - Extracted 6 random patients from `BraTS_MEN_RT` into `brats_men_rt_train.npz`.
- **Identical to v313 otherwise:** The model architecture (Float32 + SE Attention) and everything else remains strictly unchanged from v313.

## [v320] - Control Experiment: v21 Architecture + Modern Data Pipeline
*Implementation: `prompt_unet_320.py`, `p_unet_320.ipynb`*

**Research question:** Do the performance differences between v21 and later models (v292–v313) come **purely from the new vs old training data / preprocessing**?

**Identical to v21**
- **Architecture:** Filter schedule `[32, 64, 128, 256, 512]` + 1024 bottleneck. `Conv2DTranspose` decoder. No SE attention. Plain `Add()` for prompt fusion.
- **Augmentation:** 10 % probability per stage — photometric (RandomBrightness, RandomContrast, GaussianNoise), geometric (RandomFlip, RandomRotation ±5 %, RandomZoom ±5 %, RandomTranslation ±5 %), morphological (cut-out, false positives, selective erode/dilate).
- **Loss & Optimizer:** `binary_crossentropy` + `Adam` + `ExponentialDecay` (initial LR 1e-3, decay rate 0.85, staircase every 2000 epochs × steps/epoch).
- **Hyperparameters:** 3664 epochs, batch 128, dp_training 3500, dp_testing 1000, offset 12, max_number_labels 4, new\_ds every 75 epochs, validation every 300 epochs.

**Different from v21 (infrastructure only)**
- **DataGenerator:** Current `DataGenerator.py` (isotropic volumes, label-guided 128×128 patch crop, pure-numpy, fast valid-slice index via O(1) foreground pre-computation).
- **Normalization:** `universal_normalization` (CT hard-coded HU stats, MRI masked z-score with percentile clipping) — same as v292+.
- **Training data:** 3 datasets via `DataLoader_npz`: `nako_combined`, `total_seg_combined`, `msd_combined` — same as v310–313.
- **`train_step()`:** Decorated with `@tf.function` (graph mode execution).
- **`train_epoch()`:** Native `for z in train_dataset` loop (no manual `iter` / `next`).
- **Pipeline:** Persistent `tf.data` from-generator graph built once; only numpy buffer swapped on refresh.

**Interpretation**
- v320 vs v21: isolates the effect of **new data + preprocessing** (architecture/augmentation fixed).
- v320 vs v310–313: isolates the effect of **architecture + augmentation** (data fixed).

---

## [v314] - Depthwise Separable Convolutions Ablation Variant
*Implementation: `prompt_unet_314.py`, `p_unet_314.ipynb`*

**Architecture**
- **SeparableConv2D:** Replaced standard 3x3 `Conv2D` layers with Depthwise `SeparableConv2D` layers across all encoder and decoder stages to ablate spatial versus cross-channel correlation processing.
- **Identical to v313 otherwise:** Filter schedule `[48, 96, 192, 256, 384]`, SE attention enabled on prompt skips, and pure float32 training.

---

## [v313] - Float32 + SE Attention
*Implementation: `prompt_unet_313.py`, `p_unet_313.ipynb`*

**Architecture**
- **SE Attention enabled:** Squeeze-and-Excitation channel gates on all prompt skip connections — identical to v311.
- **Float32 training:** Pure `float32` throughout, plain `Adam` (no `LossScaleOptimizer`) — identical to v312.
- **Pure Conv2D everywhere:** Maintained from v310 onward.

This completes the 2×2 ablation matrix over the v310 generation:

| | No SE | SE |
|---|---|---|
| **float16** | v310 | v311 |
| **float32** | v312 | **v313** |

- v312 vs v313 isolates: **does SE attention help under float32 training?**
- v311 vs v313 isolates: **does float32 vs float16 matter when SE is present?**

Filter schedule, scale augmentation, and leakage fix are identical to v310–312:
`[48, 96, 192, 256, 384]` (~15 M trainable params).

---

## [v312] - Float32 Ablation Variant
*Implementation: `prompt_unet_312.py`, `p_unet_312.ipynb`*

**Architecture**
- **No Mixed Precision:** Disabled mixed precision training (pure `float32`). This is a direct ablation against v310 to verify if loss-scale under/overflows in float16 are causing transient instability issues when paired with heavy scale-augmentation.
- **Identical to v310 otherwise:** Uses pure `Conv2D` across all stages, maintains SE attention removal, and implements the identical scale augmentation and leakage fix.

## [v311] - SE Attention Ablation Variant
*Implementation: `prompt_unet_311.py`, `p_unet_311.ipynb`*

**Architecture**
- **SE Attention Re-enabled:** Restored the Squeeze-and-Excitation channel gating on all prompt skip connections. This provides a direct A/B ablation test against v310 to determine if SE channel-attention provides measurable benefits under the new scale-augmented data distribution.
- **Identical to v310 otherwise:** Uses pure `Conv2D` across all stages, maintains `mixed_float16` precision, and implements the identical scale augmentation (50% random quadratic crop) and leakage fix (origin anchored on `total_label_r`).

## [v310] - Pure Conv2D, No SE, Scale Augmentation + Leakage Fix
*Implementation: `prompt_unet_310.py`, `p_unet_310.ipynb`*

**Architecture**
- **No SE Attention:** Removed Squeeze-and-Excitation channel gates from all prompt skip connections. Prompt skips are now fused via a plain `Add()` layer (pre-v300 style). Motivation: ablate whether SE actually contributes under the new scale-augmented distribution.
- **Pure Conv2D:** Replaced `SeparableConv2D` at all encoder/decoder stages with standard `Conv2D`. In v300, SeparableConv was used at shallow stages 1–3; v310 removes it everywhere. Rationale: scale augmentation exposes shallow stages to both fine-grained (128×128 literal crop) and downsampled coarser textures (up to 256→128px), violating the spatial/channel independence assumption underlying separable convolutions.
- **Mixed Precision retained** — Same `mixed_float16` policy as v300. `Adam` is wrapped with `LossScaleOptimizer` for gradient stability.
- Filter schedule unchanged: `[48, 96, 192, 256, 384]` (~15 M parameters).

**DataGenerator (`DataGenerator.py`) — Scale Augmentation**
- Every call to `_extract_patch_2d` now randomly selects one of two spatial sampling modes:
  - **50 % — Literal 128×128 crop:** Preserves native scanner pixel spacing (1×1 mm). Teaches high-resolution boundary detail.
  - **50 % — Random quadratic crop → bilinear resize to 128×128:** Crop size sampled uniformly from `[128 px, min(256 px, image_size)]`. Bilinear resize for images, nearest-neighbor for masks. The 2× maximum downsample ratio ensures small structures remain ≥ ~5 px after resize, keeping them detectable by the network.
- The same random crop size and origin are applied consistently to `x`, `y`, `x_r`, `y_r`, and the UniverSeg-normalised variants (`x_u`).

**DataGenerator (`DataGenerator.py`) — Leakage Fix**
- Previously, the `_extract_patch_2d` crop bounding-box origin was computed from `total_label` (the unknown query ground-truth), causing the spatial context to be perfectly centered on the hidden target.
- The origin is now computed **only from `total_label_r`** (the Support/Prompt label).
- In an interactive-segmentation setting the clinician always provides the prompt, so anchoring the crop to the prompt label is clinically realistic **and** avoids any data leakage into the spatial sampling decision.

---

## [v301] - SE-Block Ablation
**Architecture**
- Reverted Squeeze-and-Excitation (SE) blocks from v300.
- All other v300 features (wider filters, hybrid convolutions) remain.

## [v300] - Architecture Expansion & New LR Schedule
*Implementation: `prompt_unet_300.py`*

**Architecture**
- **Wider Model:** Increased the filter schedule to `[48, 96, 192, 256, 384]` (yielding ~15M parameters).
- **Hybrid Convolutions:** Used standard `Conv2D` at deeper stages (4 & 5) instead of Separable Convolutions for better and richer cross-channel mixing at high channel counts, while keeping `SeparableConv2D` for earlier, shallower stages.
- **SE Attention:** Added Squeeze-and-Excitation channel attention on prompt skip connections.

**Training & Scheduling**
- **Three-Phase LR Schedule:** Implemented `WarmupFlatCosineDecay`:
  - *Warmup:* 50 epochs (1e-6 → 1e-3).
  - *Flat:* 1500 epochs (1e-3).
  - *Cosine Decay:* 2450 epochs (1e-3 → 1e-5).
- **Dataset Refresh:** Reduced `new_ds` cadence from 75 to 50 epochs to prevent overfitting and refresh data exposure.

**Augmentation Tuning**
- **Reduced Probabilities:** Reduced `prob_geometric` to 0.50 and `prob_morph` to 0.30 (which additionally provides CPU optimization).
- **Gamma Range:** Narrowed gamma augmentation range to `(0.85, 1.25)` to prevent distribution drift from the normalized z-score space.

## [v292] - Preprocessing & Datagen Overhaul
*Implementation: `prompt_unet_292.py`, `train.py`*

**Preprocessing**
- **Isotropic Resampling:** Applied to volumes prior to normalization.
- **Z-Score Normalization:** 
  - *CT:* Hardcoded stats (Mean: -15, Std: 160) with clipping between `[-1000, 1000]`.
  - *MRI:* Foreground-only statistics with 0.5% – 99.5% percentile clipping.
- **Data Loading:** Switched to efficient `.npz` based data loaders rather than legacy structures.

**Data Generator & Stability**
- **OOM Fix & Pure NumPy:** The entire data processing pipeline was rewritten (`DataGenerator.py`) to map exclusively to NumPy arrays across the CPU, directly returning stacked arrays to bypass TensorFlow graph node registration per-call, mitigating severe GPU/CPU memory fragmentation and Out-of-Memory errors.
- **2D Patch-Based Extraction:** Instead of performing random 3D volume crops or scaling images linearly, slices are extracted as **exact 128x128 patches**. This uses label-guided logic to ensure patches capture the target structure while maintaining a 1:1 pixel ratio (no rescaling). Dimensions smaller than 128 are symmetrically padded (`-5.0` for images, `0.0` for masks) to prevent spatial distortion.
- **Processing & Caching:** Generates implement a swift per-call normalization cache (`_norm_cache`) resolving redundant volume operations for matched patients, additionally tagging explicit output modalities (`m_np` mapped `0.0 = CT`, `1.0 = MRI`).
- **Filtering:** Samples falling below defined axis thresholds are comprehensively ignored to prevent artifact induction.

**Codebase Refactoring**
- Refactored logic out of notebooks into standalone `.py` files (`train.py`, `optimizer.py`, model definitions) while keeping notebooks solely for experimentation and evaluation pipelines.

## [v282 / v283] - Efficiency & Optimization
*Implementation: `p_unet_282.ipynb`, `p_unet_283.ipynb`*

**Efficiency Improvements**
- **Mixed Precision:** Enabled `mixed_float16` training global policy.
- **Speed Improvements:** Switched to Depthwise `SeparableConv2D` globally reducing computation costs, and optimized graph executions.

**Architecture Tweaks**
- Built an **Asymmetric U-Net** structure, using a lighter decoder to save parameters while maintaining representational capacity.
- **Thinned Bottleneck:** Reduced parameter weight in the model bottleneck.
- **Upsampling Changes:** Replaced generic `Conv2DTranspose` configurations with math-based scaling and lighter alternatives (e.g. `SeparableConv2D` followed by UpSampling).

**Training & Scheduling**
- **New Scheduler (v282):** Implemented Keras 3 `CosineDecay` with built-in warmup to stabilize mixed precision training gradients:
  - *Initial LR:* `1e-6`
  - *Warmup:* 50 epochs ramping up to `1e-3` (peak).
  - *Decay Phase:* Smooth cosine curve down over 4950 epochs to a final LR of `1e-5` (1% of peak).
- **v283 Specific:** Reverted to the older `ExponentialDecay` model scheduler version.

## [v272] - Data Distribution Tuning
*Implementation: `p_unet_272.ipynb`*

**Dataset Composition**
- Added CT dataset (Total Segmentor) (45 unique CT, 61 MRI).

**Augmentation Pipeline Details**
- Removed double blurring effects.
- Increased the strength parameter of Gaussian blur.
- Implemented a significant overall increase in general photometric and geometric augmentation probabilities.

## [v21] - Integration Phase (Legacy Architecture)
*Implementation: `p_unet_21.ipynb`*

**History & Goal**
- Merged the augmentation strategy from v1.6.5 with the network structure additions of v2.0 (Dropout). Focused training restricted strictly to the NAKO dataset.

**Architecture (Dual-Encoder Prompt-UNet)**
- **Filter Schedule:** Followed traditional uniform multiplier progression `[32, 64, 128, 256, 512]` via standard `Conv2D`. 
- **Prompt Encoder:** Separate 5-stage feature extraction encoding path mapping the two-channel prompt mask context.
- **Image Input Encoder:** Main 5-stage sequential encoder for the target image. It featured a conditioning mechanism adding (`layers.Add()`) prompt feature maps to its own feature abstractions at each stage.
- **Bottleneck:** Contained an expanded 1024 filter depth layout with increased `Dropout` (0.2 parameter compared to 0.1 in shallower tiers).
- **Decoder:** Emphasized symmetric transposed convolutions (`Conv2DTranspose`) mapped back to `[512, 256, 128, 64, 32]` while concatenating features originating from the input image encoder path. 
- **Layers Structure:** Relied on `BatchNormalization` followed directly by `LeakyReLU`.

**Training & Schedulers**
- **Loss:** `binary_crossentropy`.
- Introduced Training Schedulers: relied on TensorFlow's built-in `ExponentialDecay`. Learning rate began at `0.001` fading out at a decay rate of `0.85`, configured systematically to run every `2000` steps (`decay_steps=steps_per_epoch*decay_epochs`). 

**Data & Augmentation Strategy**
- **Legacy Datagen Framework (`DataGenerator_old.py`):** Structured fundamentally using `tf.data.Dataset` arrays running nested mapping instances inside `tf.function` iterations recursively.
  - Slices persistently cached `tf.tensors` natively inside its core loops dynamically driving Out-of-Memory spikes across prolonged generation tasks.
  - **Volume Cropping & Rescaling:** Samples were generated by first taking a **random 3D volume crop** (of arbitrary size) from the original scan and then applying **Nearest-Neighbor Resizing** (`tf.image.resize`) to force the resulting slices into $128 \times 128$. This approach introduced significant spatial distortion as anatomy was squashed or stretched depending on the initial crop dimensions.
  - Implemented variable slice distance offsets mimicking prompt trace variance.
- Augmentations operated at a low application baseline (10% standard chance `0.1`), configured separately per inputs:
  - **Photometric:** Applied strictly to target image (`x`): RandomBrightness, RandomContrast, Lambda-injected noise, and GaussianNoise.
  - **Geometric:** Applied coherently across spatial domains (`x, y, p`): RandomFlip, RandomRotation (with `reflect` fill and `nearest` interpolation), RandomZoom, RandomTranslation. 
  - **Prompt Morphological Distortions:** Applied rigorously to simulate human error on the `prompt (p)`:
    - *Cut-out:* Selectively clearing max 20% fraction of positive prompt slices to spoof unnoted areas.
    - *False Positives:* Spraying random positive masks outside target areas.
    - *Morph:* Emphasizing selective erosion & dilation via custom tensor map injections restricted cleanly by structure size bounding (`min_size=30`) and scaled kernel sizes `max_kernel=2`.
