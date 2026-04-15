"""
DataGenerator — pure numpy data pipeline for Prompt U-Net training (and testing).

All tensor operations happen in numpy on the CPU.  No TensorFlow ops are
used inside this module, which means:
  - No GPU memory is touched during data generation.
  - No tf.data graph nodes are registered per call.
  - Python GC manages all memory predictably.

The public interface produces stacked numpy arrays that the training loop
feeds into a persistent tf.data.from_generator pipeline (built once).
"""

import random
import time

import numpy as np
from scipy.ndimage import label as scipy_label

from utils.preprocessing import universal_normalization


class DataGenerator:
    """
    Generates (image, label, prompt, modality) tuples from a DataLoader.

    Parameters
    ----------
    dataloader    : DataLoader object whose .dataset[id] returns a dict with
                    keys 'image', 'segmentations', and optionally 'modality'.
    img_height    : Target patch height (default 128).
    img_width     : Target patch width  (default 128).
    minimum_pixel : Minimum foreground pixels required in each mask slice.
    """

    def __init__(self, dataloader, img_height=128, img_width=128, minimum_pixel=5):
        self.dataloader    = dataloader
        self.height        = img_height
        self.width         = img_width
        self.minimum_pixel = minimum_pixel

        # Per-call normalization cache: pid → (x_norm_np, segs)
        # Cleared at the start of every get_data_points* call.
        self._norm_cache: dict = {}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_2d_slice(self, vol, idx, axis):
        """Return a 2-D numpy slice from a 3-D volume along `axis`."""
        if axis == 'x':
            return vol[idx, :, :]
        elif axis == 'y':
            return vol[:, idx, :]
        else:  # 'z'
            return vol[:, :, idx]

    def _prepare_volume(self, current_dict, pid=None):
        """
        Normalize the image for one patient and return (x_np, segs).
        Results are cached by `pid` for the duration of one data-generation call.
        """
        if pid is not None and pid in self._norm_cache:
            return self._norm_cache[pid]

        segs     = current_dict['segmentations']
        x        = np.asarray(current_dict['image'], dtype=np.float32)
        modality = current_dict.get('modality', 'UNKNOWN')

        x = universal_normalization(x, modality=modality)  # returns np.float32

        result = (x, segs)
        if pid is not None:
            self._norm_cache[pid] = result
        return result

    def _pad_if_needed(self, arr, target, pad_val=0.0):
        """Zero-pad a 2-D array symmetrically so both dims >= target."""
        h, w   = arr.shape
        pad_h  = max(0, target - h)
        pad_w  = max(0, target - w)
        top    = pad_h // 2;  bot  = pad_h - top
        left   = pad_w // 2;  right = pad_w - left
        return np.pad(arr, ((top, bot), (left, right)),
                      mode='constant', constant_values=pad_val)

    def _extract_patch_2d(self, x_2d, total_label, x_2d_r, total_label_r):
        """
        Label-guided 128×128 crop of a 2-D slice pair.

        The crop window is guaranteed to overlap with the bounding box of
        `total_label`.  The reference slice is cropped at the same position.

        Returns
        -------
        x_2d, x_2d_r        : shape (128, 128, 1) float32
        total_label, total_label_r : shape (128, 128, 1) float32
        """
        ps = self.height  # patch size (128)

        x_2d        = self._pad_if_needed(x_2d,        ps, pad_val=-5.0)
        x_2d_r      = self._pad_if_needed(x_2d_r,      ps, pad_val=-5.0)
        total_label = self._pad_if_needed(total_label,  ps, pad_val=0.0)
        total_label_r = self._pad_if_needed(total_label_r, ps, pad_val=0.0)

        h, w = x_2d.shape

        # --- Label-guided crop origin ---
        nonzero = np.argwhere(total_label > 0)
        if len(nonzero) > 0:
            min_h_l, min_w_l = nonzero.min(axis=0)
            max_h_l, max_w_l = nonzero.max(axis=0)

            lo_h = int(max(0,          max_h_l - ps + 1))
            hi_h = int(min(h - ps,     min_h_l))
            hi_h = max(lo_h, hi_h)

            lo_w = int(max(0,          max_w_l - ps + 1))
            hi_w = int(min(w - ps,     min_w_l))
            hi_w = max(lo_w, hi_w)

            sh = random.randint(lo_h, hi_h)
            sw = random.randint(lo_w, hi_w)
        else:
            sh = random.randint(0, max(0, h - ps))
            sw = random.randint(0, max(0, w - ps))

        def crop(a):
            return a[sh:sh + ps, sw:sw + ps]

        x_2d        = crop(x_2d)
        x_2d_r      = crop(x_2d_r)
        total_label = crop(total_label)
        total_label_r = crop(total_label_r)

        # Add channel dim → (128, 128, 1)
        return (
            x_2d[..., np.newaxis].astype(np.float32),
            total_label[..., np.newaxis].astype(np.float32),
            x_2d_r[..., np.newaxis].astype(np.float32),
            total_label_r[..., np.newaxis].astype(np.float32),
        )

    def _sample_offset(self, i, offset, length):
        """Return a random signed offset ±[1..offset] that stays in-bounds."""
        possible = list(range(-offset, 0)) + list(range(1, offset + 1))
        r = random.choice(possible)
        if (i + r < 0) or (i + r >= length):
            return None
        return r

    def _select_valid_labels(self, y_2d, y_2d_r, max_number_labels, primary_task):
        """
        Build binary label lists for the primary and up to (max_number_labels-1)
        secondary structures.

        Supports two segmentation formats:
          - List of binary channel arrays  (multi-channel)
          - Single integer label volume    (multi-label int)

        Returns (labels_out, labels_r_out) as lists of 2-D numpy arrays,
        or (None, None) if the primary task is absent / too small.
        """
        count = random.randint(1, max_number_labels)
        labels_out, labels_r_out = [], []

        is_multi_channel = isinstance(y_2d, list) and len(y_2d) > 1

        if isinstance(y_2d, list):
            if len(y_2d) == 0:
                return None, None
            if len(y_2d) == 1:
                # Unwrap single-element list → treat as integer volume
                y_2d   = np.asarray(y_2d[0])
                y_2d_r = np.asarray(y_2d_r[0])
                is_multi_channel = False
            else:
                y_2d   = [np.asarray(s) for s in y_2d]
                y_2d_r = [np.asarray(s) for s in y_2d_r]
        else:
            y_2d   = np.asarray(y_2d)
            y_2d_r = np.asarray(y_2d_r)

        # ---- CASE 1: Multi-channel list ----
        if is_multi_channel:
            if primary_task is not None:
                idx = primary_task - 1
                s1 = (y_2d[idx]   > 0).astype(np.float32)
                s2 = (y_2d_r[idx] > 0).astype(np.float32)
                if s1.sum() < self.minimum_pixel or s2.sum() < self.minimum_pixel:
                    return None, None
                labels_out.append(s1)
                labels_r_out.append(s2)

                if count > 1:
                    others = []
                    for other_idx, (seg, seg_r) in enumerate(zip(y_2d, y_2d_r)):
                        if other_idx == idx:
                            continue
                        o1 = (seg   > 0).astype(np.float32)
                        o2 = (seg_r > 0).astype(np.float32)
                        if o1.sum() >= self.minimum_pixel and o2.sum() >= self.minimum_pixel:
                            others.append((o1, o2))
                    random.shuffle(others)
                    for o1, o2 in others[:count - 1]:
                        labels_out.append(o1)
                        labels_r_out.append(o2)

        # ---- CASE 2: Integer label volume ----
        else:
            if primary_task is not None:
                label   = (y_2d   == primary_task).astype(np.float32)
                label_r = (y_2d_r == primary_task).astype(np.float32)
                if label.sum() < self.minimum_pixel or label_r.sum() < self.minimum_pixel:
                    return None, None
                labels_out.append(label)
                labels_r_out.append(label_r)

                if count > 1:
                    vals_1 = set(np.unique(y_2d).tolist())
                    vals_2 = set(np.unique(y_2d_r).tolist())
                    vals_1.discard(0); vals_1.discard(primary_task)
                    vals_2.discard(0); vals_2.discard(primary_task)
                    candidates = list(vals_1 & vals_2)
                    random.shuffle(candidates)

                    for lv in candidates:
                        l   = (y_2d   == lv).astype(np.float32)
                        l_r = (y_2d_r == lv).astype(np.float32)
                        if l.sum() < self.minimum_pixel or l_r.sum() < self.minimum_pixel:
                            continue
                        labels_out.append(l)
                        labels_r_out.append(l_r)
                        if len(labels_out) == count:
                            break

        if not labels_out:
            return None, None
        return labels_out, labels_r_out

    def _merge_labels(self, labels):
        """Sum a list of binary 2-D arrays into one merged mask."""
        result = labels[0].copy()
        for l in labels[1:]:
            result = result + l
        return result

    def _create_single_datapoint(self, x, y, i, r, d, max_number_labels, primary_task):
        """
        Attempt to build one (x_2d, y_2d, prompt) tuple.

        Returns None if the candidate slice pair doesn't pass quality checks.
        All arrays are float32 numpy, shapes (128,128,1) / (128,128,2).
        """
        if isinstance(y, list):
            y_2d   = [self._get_2d_slice(np.asarray(seg), i,   d) for seg in y]
            y_2d_r = [self._get_2d_slice(np.asarray(seg), i+r, d) for seg in y]
        else:
            y_2d   = self._get_2d_slice(y, i,   d)
            y_2d_r = self._get_2d_slice(y, i+r, d)

        labels, labels_r = self._select_valid_labels(y_2d, y_2d_r, max_number_labels, primary_task)
        if labels is None:
            return None

        total_label   = self._merge_labels(labels)
        total_label_r = self._merge_labels(labels_r)

        x_2d   = self._get_2d_slice(x, i,   d)
        x_2d_r = self._get_2d_slice(x, i+r, d)

        x_2d, total_label, x_2d_r, total_label_r = self._extract_patch_2d(
            x_2d, total_label, x_2d_r, total_label_r
        )

        # Post-crop quality check: both masks must be visible after cropping
        if (np.count_nonzero(total_label)   < self.minimum_pixel or
                np.count_nonzero(total_label_r) < self.minimum_pixel):
            return None

        # Prompt = [reference_image_slice | reference_label_slice] stacked on C
        p = np.concatenate([x_2d_r, total_label_r], axis=-1)  # (128,128,2)
        return x_2d, total_label, p  # all (128,128,1/2) float32

    def _process_dimension(self, x, y, d, offset, max_number_labels,
                           x_new, y_new, prompt, offset_list, m_new,
                           modality_flag, slices_added, max_data_points):
        """
        Attempt to harvest data points along axis `d` for one patient volume.
        Returns updated `slices_added` count.
        """
        # Skip dimension if any in-plane extent is smaller than the patch size.
        dim_idx    = 'xyz'.index(d)
        slice_dims = [s for i, s in enumerate(x.shape) if i != dim_idx]
        if min(slice_dims) < max(self.height, self.width):
            return slices_added

        if isinstance(y, list):
            y_shape    = np.asarray(y[0]).shape[dim_idx]
            valid_tasks = list(range(1, len(y) + 1))
        else:
            y_shape = y.shape[dim_idx]
            flat    = y.reshape(-1)
            valid_tasks = [int(v) for v in np.unique(flat) if v != 0]

        if not valid_tasks:
            return slices_added

        slices_added_per_pid = 0
        failed_searches      = 0

        while (slices_added_per_pid < 150
               and slices_added      < max_data_points
               and failed_searches   < 100):

            primary_task  = random.choice(valid_tasks)
            slice_indices = list(range(y_shape))
            random.shuffle(slice_indices)

            found = False
            for i in slice_indices:
                r = self._sample_offset(i, offset, y_shape)
                if r is None:
                    continue

                result = self._create_single_datapoint(
                    x, y, i, r, d, max_number_labels, primary_task
                )
                if result is not None:
                    x2d, y2d, p = result
                    x_new.append(x2d)
                    y_new.append(y2d)
                    prompt.append(p)
                    offset_list.append(r)
                    m_new.append(modality_flag)
                    slices_added         += 1
                    slices_added_per_pid += 1
                    failed_searches       = 0
                    found = True
                    break  # pick a new task on the next iteration

            if not found:
                failed_searches += 1

        return slices_added

    def _randomize(self, x_new, y_new, prompt, offset_list, m_new):
        """Shuffle all lists in unison."""
        combined = list(zip(x_new, y_new, prompt, offset_list, m_new))
        random.shuffle(combined)
        if combined:
            x_new, y_new, prompt, offset_list, m_new = zip(*combined)
            return (list(x_new), list(y_new), list(prompt),
                    list(offset_list), list(m_new))
        return x_new, y_new, prompt, offset_list, m_new

    # ------------------------------------------------------------------ #
    #  Core generation loop                                                #
    # ------------------------------------------------------------------ #

    def _collect_data_points(self, max_data_points, offset, max_number_labels, dimensions):
        """
        Internal workhorse.  Returns five parallel lists of numpy arrays
        (x, y, prompt, offsets, modality_flags).
        """
        start = time.time()
        print("Creating new Data Points ...")

        self._norm_cache.clear()

        x_new, y_new, prompt, offset_list, m_new = [], [], [], [], []
        slices_added = 0

        while slices_added < max_data_points:
            random.shuffle(self.dataloader.current_ids)

            for pid in self.dataloader.current_ids:
                current_dict = self.dataloader.dataset[pid]
                modality_str = current_dict.get('modality', 'UNKNOWN')
                is_mri       = 0.0 if modality_str == 'CT' else 1.0

                x, y = self._prepare_volume(current_dict, pid=pid)

                # Unwrap single-element list
                if isinstance(y, list) and len(y) == 1:
                    y = np.asarray(y[0])

                dims = list(dimensions)
                random.shuffle(dims)

                for d in dims:
                    if slices_added >= max_data_points:
                        break

                    slices_added = self._process_dimension(
                        x, y, d, offset, max_number_labels,
                        x_new, y_new, prompt, offset_list, m_new,
                        is_mri, slices_added, max_data_points
                    )

                if slices_added >= max_data_points:
                    break

        x_new, y_new, prompt, offset_list, m_new = self._randomize(
            x_new, y_new, prompt, offset_list, m_new
        )
        print(f'It took {time.time() - start:.0f} seconds')
        return x_new, y_new, prompt, offset_list, m_new

    def _to_numpy_arrays(self, x_lst, y_lst, p_lst, m_lst):
        """Stack lists of (128,128,C) arrays into (N,128,128,C) float32 arrays."""
        return (
            np.stack(x_lst).astype(np.float32),
            np.stack(y_lst).astype(np.float32),
            np.stack(p_lst).astype(np.float32),
            np.array(m_lst, dtype=np.float32),
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_data_points_numpy(self, max_data_points=3500, offset=5,
                              max_number_labels=1, dimensions=None):
        """
        Generate training data and return stacked numpy arrays.

        Returns
        -------
        x_np   : (N, 128, 128, 1)  float32
        y_np   : (N, 128, 128, 1)  float32
        p_np   : (N, 128, 128, 2)  float32
        m_np   : (N,)              float32  — 0.0 = CT, 1.0 = MRI
        offsets: list[int]
        """
        if dimensions is None:
            dimensions = ['x', 'y', 'z']
        self.dataloader.current_ids = self.dataloader.train_ids
        x, y, p, off, m = self._collect_data_points(
            max_data_points, offset, max_number_labels, dimensions
        )
        x_np, y_np, p_np, m_np = self._to_numpy_arrays(x, y, p, m)
        return x_np, y_np, p_np, m_np, off

    def get_val_data_points_numpy(self, max_data_points=3500, offset=5,
                                  max_number_labels=1, dimensions=None):
        """Validation variant of get_data_points_numpy."""
        if dimensions is None:
            dimensions = ['x', 'y', 'z']
        self.dataloader.current_ids = self.dataloader.validation_ids
        x, y, p, off, m = self._collect_data_points(
            max_data_points, offset, max_number_labels, dimensions
        )
        x_np, y_np, p_np, m_np = self._to_numpy_arrays(x, y, p, m)
        return x_np, y_np, p_np, m_np, off

    # ---- Legacy tf.data wrappers (kept for backward compatibility) ---- #

    def get_data_points(self, max_data_points=3500, offset=5,
                        max_number_labels=1, dimensions=None):
        """
        Returns a tf.data.Dataset of (x, y, prompt, modality) tuples.
        Prefer get_data_points_numpy() for training to avoid graph bloat.
        """
        import tensorflow as tf
        if dimensions is None:
            dimensions = ['x', 'y', 'z']
        x_np, y_np, p_np, m_np, off = self.get_data_points_numpy(
            max_data_points, offset, max_number_labels, dimensions
        )
        ds = tf.data.Dataset.from_tensor_slices((x_np, y_np, p_np, m_np))
        return ds, off

    def get_val_data_points(self, max_data_points=3500, offset=5,
                            max_number_labels=1, dimensions=None):
        """Validation variant of get_data_points (tf.data wrapper)."""
        import tensorflow as tf
        if dimensions is None:
            dimensions = ['x', 'y', 'z']
        x_np, y_np, p_np, m_np, off = self.get_val_data_points_numpy(
            max_data_points, offset, max_number_labels, dimensions
        )
        ds = tf.data.Dataset.from_tensor_slices((x_np, y_np, p_np, m_np))
        return ds, off