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

from utils.preprocessing import universal_normalization, universeg_normalization


class DataGenerator:
    """
    Generates (image, label, prompt, modality) tuples from a DataLoader.

    RAM Management Note:
    The DataGenerator evaluates CPU-intensive float32 scaling operations (Z-score, UniverSeg) 
    per patient and permanently caches the results in `_norm_cache` and `_universeg_cache`. 
    These caches persist indefinitely across your entire program execution to preserve sampling speed 
    across multiple generator loops. 
    CONSEQUENCE: If you run through a massive dataset, these duplicate `float32` arrays will stack 
    in RAM alongside your existing DataLoader raw arrays! If your memory is maxing out, you can 
    manually call `gen._norm_cache.clear()` and `gen._universeg_cache.clear()` sequentially in 
    your pipeline to release memory. Note that doing so forces any future samples from those patients 
    to be re-scaled and re-evaluated, which will slow training down.

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
        # Per-call UniverSeg normalization cache: pid → x_u_np
        self._universeg_cache: dict = {}
        # Per-call valid-slice index: (pid, dim_idx) → {task_id: np.ndarray of valid slice indices}
        # Avoids O(y_shape) scans for sparse structures (e.g. small body organs).
        self._slice_index_cache: dict = {}

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

    def _prepare_universeg_volume(self, current_dict, pid=None):
        """
        Apply UniverSeg normalization to the raw volume and cache by pid.
        Reads current_dict['image'] directly to get RAW intensities before
        our z-score is applied — call alongside _prepare_volume.
        """
        if pid is not None and pid in self._universeg_cache:
            return self._universeg_cache[pid]

        x        = np.asarray(current_dict['image'], dtype=np.float32)
        modality = current_dict.get('modality', 'UNKNOWN')
        x_u      = universeg_normalization(x, modality=modality)

        if pid is not None:
            self._universeg_cache[pid] = x_u
        return x_u

    def _build_valid_slice_index(self, y, dim_idx):
        """
        Pre-compute which slices along `dim_idx` contain at least
        self.minimum_pixel foreground pixels for each task.

        Parameters
        ----------
        y       : list of 3-D arrays (multi-channel) or single 3-D int array
        dim_idx : 0='x', 1='y', 2='z'

        Returns
        -------
        dict : task_id (int, 1-based for multi-channel; label value for int volume)
               → np.ndarray of valid slice indices along dim_idx
        """
        valid    = {}
        in_plane = tuple(ax for ax in range(3) if ax != dim_idx)

        if isinstance(y, list):
            for ch_idx, seg in enumerate(y):
                arr    = np.asarray(seg, dtype=np.float32)
                counts = arr.sum(axis=in_plane)           # shape: (num_slices,)
                idx    = np.where(counts >= self.minimum_pixel)[0]
                if idx.size > 0:
                    valid[ch_idx + 1] = idx               # 1-based task id
        else:
            y_arr = np.asarray(y)
            for lv in np.unique(y_arr):
                lv = int(lv)
                if lv == 0:
                    continue
                counts = (y_arr == lv).sum(axis=in_plane)
                idx    = np.where(counts >= self.minimum_pixel)[0]
                if idx.size > 0:
                    valid[lv] = idx

        return valid

    def _get_slice_index(self, y, pid, dim_idx):
        """
        Return (and lazily build + cache) the valid-slice index for (pid, dim_idx).
        When pid is None the index is built on the fly without caching.
        """
        if pid is None:
            return self._build_valid_slice_index(y, dim_idx)
        key = (pid, dim_idx)
        if key not in self._slice_index_cache:
            self._slice_index_cache[key] = self._build_valid_slice_index(y, dim_idx)
        return self._slice_index_cache[key]

    def _pad_if_needed(self, arr, target, pad_val=0.0):
        """Zero-pad a 2-D array symmetrically so both dims >= target."""
        h, w   = arr.shape
        pad_h  = max(0, target - h)
        pad_w  = max(0, target - w)
        top    = pad_h // 2;  bot  = pad_h - top
        left   = pad_w // 2;  right = pad_w - left
        return np.pad(arr, ((top, bot), (left, right)),
                      mode='constant', constant_values=pad_val)

    def _extract_patch_2d(self, x_2d, total_label, x_2d_r, total_label_r,
                           x_u_2d=None, x_u_2d_r=None):
        """
        Label-guided crop of a 2-D slice pair with random Scale Augmentation.
        
        50% chance to extract exactly 128x128 crop.
        50% chance to extract a random quadratic crop (e.g. 150x150 up to 256x256) 
        and mathematically resize it down to 128x128 for scale invariance.
        
        The crop window is strictly guaranteed to overlap with the bounding box of
        `total_label_r` (the Support/Prompt mask)
        """
        import PIL.Image as Image
        
        orig_h, orig_w = x_2d.shape
        max_bound = min(orig_h, orig_w)
        
        # 50% chance for random quadratic resolution (scale jitter)
        if random.random() < 0.5 and max_bound > self.height:
            upper_bound = min(256, max_bound)
            if upper_bound > self.height:
                ps = random.randint(self.height, upper_bound)
            else:
                ps = self.height
        else:
            ps = self.height

        x_2d          = self._pad_if_needed(x_2d,          ps, pad_val=-5.0)
        x_2d_r        = self._pad_if_needed(x_2d_r,        ps, pad_val=-5.0)
        total_label   = self._pad_if_needed(total_label,   ps, pad_val=0.0)
        total_label_r = self._pad_if_needed(total_label_r, ps, pad_val=0.0)

        # pad_val=0.0: background is 0 in UniverSeg's [0, 1] space
        if x_u_2d is not None:
            x_u_2d   = self._pad_if_needed(x_u_2d,   ps, pad_val=0.0)
            x_u_2d_r = self._pad_if_needed(x_u_2d_r, ps, pad_val=0.0)

        h, w = x_2d.shape

        # --- Label-guided crop origin (Origin must only be calculated from the Support Prompt (total_label_r)) ---
        nonzero = np.argwhere(total_label_r > 0)
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

        def crop_and_resize(a, is_mask=False):
            patch = a[sh:sh + ps, sw:sw + ps]
            if ps == self.height:  # No resizing needed
                return patch
            resample_mode = Image.NEAREST if is_mask else Image.BILINEAR
            patch_img = Image.fromarray(patch)
            
            # Note: PIL resize takes (width, height)
            resized_img = patch_img.resize((self.width, self.height), resample=resample_mode)
            return np.array(resized_img, dtype=a.dtype)

        x_2d_res        = crop_and_resize(x_2d,        is_mask=False)
        x_2d_r_res      = crop_and_resize(x_2d_r,      is_mask=False)
        total_label_res = crop_and_resize(total_label, is_mask=True)
        total_label_r_res = crop_and_resize(total_label_r, is_mask=True)

        # Add channel dim → (128, 128, 1)
        result = (
            x_2d_res[..., np.newaxis].astype(np.float32),
            total_label_res[..., np.newaxis].astype(np.float32),
            x_2d_r_res[..., np.newaxis].astype(np.float32),
            total_label_r_res[..., np.newaxis].astype(np.float32),
        )

        if x_u_2d is not None:
            x_u_2d_res   = crop_and_resize(x_u_2d,   is_mask=False)
            x_u_2d_r_res = crop_and_resize(x_u_2d_r, is_mask=False)
            return result + (
                x_u_2d_res[..., np.newaxis].astype(np.float32),
                x_u_2d_r_res[..., np.newaxis].astype(np.float32),
            )
        return result

    def _extract_fullslice_2d(self, x_2d, total_label, x_2d_r, total_label_r,
                               x_u_2d=None, x_u_2d_r=None):
        """
        Extract the maximum centered square from each 2D slice and resize it
        to (self.height × self.width).  Unlike _extract_patch_2d, no
        label-guided crop offset is applied — the full anatomical context of
        the cross-section is always preserved.

        Example: a 278×290 axial slice → center-crop to 278×278 → resize to
        128×128.  Works for any in-plane size, including slices smaller than
        self.height (they are upscaled), so no in-plane size guard is needed.
        """
        import PIL.Image as Image

        h, w = x_2d.shape
        sq   = min(h, w)   # largest square that fits in this cross-section

        top  = (h - sq) // 2
        left = (w - sq) // 2

        def crop_and_resize(a, is_mask=False):
            patch = a[top:top + sq, left:left + sq]
            if sq == self.height and self.height == self.width:
                return patch   # already the right size, skip PIL round-trip
            resample = Image.NEAREST if is_mask else Image.BILINEAR
            return np.array(
                Image.fromarray(patch).resize((self.width, self.height), resample=resample),
                dtype=a.dtype,
            )

        x_r    = crop_and_resize(x_2d,         is_mask=False)
        xref_r = crop_and_resize(x_2d_r,       is_mask=False)
        y_r    = crop_and_resize(total_label,   is_mask=True)
        yref_r = crop_and_resize(total_label_r, is_mask=True)

        result = (
            x_r[..., np.newaxis].astype(np.float32),
            y_r[..., np.newaxis].astype(np.float32),
            xref_r[..., np.newaxis].astype(np.float32),
            yref_r[..., np.newaxis].astype(np.float32),
        )

        if x_u_2d is not None:
            xu_r   = crop_and_resize(x_u_2d,   is_mask=False)
            xu_ref = crop_and_resize(x_u_2d_r, is_mask=False)
            return result + (
                xu_r[..., np.newaxis].astype(np.float32),
                xu_ref[..., np.newaxis].astype(np.float32),
            )
        return result

    def _extract_native_2d(self, x_2d, total_label, x_2d_r, total_label_r,
                           x_u_2d=None, x_u_2d_r=None):
        """
        Extract the full cross-section exactly as it appears in the origin volume
        without any uniform reshaping, cropping, or interpolation logic. This preserves 
        the native aspect ratio and image sizes exactly for downstream tiling architectures.
        """
        result = (
            x_2d[..., np.newaxis].astype(np.float32),
            total_label[..., np.newaxis].astype(np.float32),
            x_2d_r[..., np.newaxis].astype(np.float32),
            total_label_r[..., np.newaxis].astype(np.float32),
        )
        if x_u_2d is not None:
            return result + (
                x_u_2d[..., np.newaxis].astype(np.float32),
                x_u_2d_r[..., np.newaxis].astype(np.float32),
            )
        return result

    def _sample_offset(self, i, offset, length):
        """Return a random signed offset ±[1..offset] that stays in-bounds.

        Filters to only valid offsets before sampling, so this never returns
        None when valid offsets exist (e.g. for boundary slices where negative
        offsets would be out-of-bounds, only positive ones are considered).
        """
        possible = [r for r in list(range(-offset, 0)) + list(range(1, offset + 1))
                    if 0 <= i + r < length]
        if not possible:
            return None
        return random.choice(possible)

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

    def _create_single_datapoint(self, x, y, i, r, d, max_number_labels, primary_task,
                                  x_u=None, extraction_mode='crop'):
        """
        Attempt to build one (x_2d, y_2d, prompt[, x_u_2d]) tuple.

        Returns None if the candidate slice pair doesn't pass quality checks.
        All arrays are float32 numpy, shapes (128,128,1) / (128,128,2).

        If `x_u` (UniverSeg-normalised volume) is provided, the same spatial
        transform is applied to extract `x_u_2d` (128,128,1) in [0, 1] and it
        is appended to the return tuple:  (x_2d, y_2d, prompt, x_u_2d).

        Parameters
        ----------
        extraction_mode : 'crop' (default), 'fullslice', or 'native'
            'crop'      → label-guided patch crop with optional scale jitter
                          (_extract_patch_2d). Requires in-plane dims >= 128.
            'fullslice' → maximum centered square resized to target resolution
                          (_extract_fullslice_2d). Works for any in-plane size.
            'native'    → keeps exact native slice shape matching the source volume.
                          (_extract_native_2d). Returns variable-sized arrays.
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

        # Extract UniverSeg slices at the SAME positions before the shared crop
        x_u_2d   = self._get_2d_slice(x_u, i,   d) if x_u is not None else None
        x_u_2d_r = self._get_2d_slice(x_u, i+r, d) if x_u is not None else None

        if extraction_mode == 'fullslice':
            patch = self._extract_fullslice_2d(
                x_2d, total_label, x_2d_r, total_label_r,
                x_u_2d, x_u_2d_r,
            )
        elif extraction_mode == 'native':
            patch = self._extract_native_2d(
                x_2d, total_label, x_2d_r, total_label_r,
                x_u_2d, x_u_2d_r,
            )
        else:
            patch = self._extract_patch_2d(
                x_2d, total_label, x_2d_r, total_label_r,
                x_u_2d, x_u_2d_r,
            )

        if x_u is not None:
            x_2d, total_label, x_2d_r, total_label_r, x_u_2d, x_u_2d_r = patch
        else:
            x_2d, total_label, x_2d_r, total_label_r = patch

        # Post-crop quality check: both masks must be visible after cropping
        if (np.count_nonzero(total_label)   < self.minimum_pixel or
                np.count_nonzero(total_label_r) < self.minimum_pixel):
            return None

        # Prompt = [reference_image_slice | reference_label_slice] stacked on C
        p = np.concatenate([x_2d_r, total_label_r], axis=-1)  # (128,128,2)

        if x_u is not None:
            return x_2d, total_label, p, x_u_2d   # x_u_2d_r not stored (not used by any model)
        return x_2d, total_label, p  # all (128,128,1/2) float32

    def _process_dimension(self, x, y, d, offset, max_number_labels,
                           x_new, y_new, prompt, offset_list, m_new,
                           modality_flag, slices_added, max_data_points, pid=None):
        """
        Attempt to harvest data points along axis `d` for one patient volume.
        Returns updated `slices_added` count.

        Uses a pre-computed valid-slice index (cached per pid) so that each
        outer-loop iteration is O(1) — sampling directly from slices that are
        known to contain foreground — instead of O(y_shape) blind shuffling.
        This eliminates 30-second stalls for sparse body structures.
        """
        # Skip dimension if any in-plane extent is smaller than the patch size.
        dim_idx    = 'xyz'.index(d)
        slice_dims = [s for ax, s in enumerate(x.shape) if ax != dim_idx]
        if min(slice_dims) < max(self.height, self.width):
            return slices_added

        if isinstance(y, list):
            y_shape = np.asarray(y[0]).shape[dim_idx]
        else:
            y_shape = y.shape[dim_idx]

        # Build (or fetch cached) valid-slice index for this (pid, dim_idx).
        # valid_index: task_id → np.ndarray of slice indices with ≥ minimum_pixel foreground.
        valid_index = self._get_slice_index(y, pid, dim_idx)
        valid_tasks = list(valid_index.keys())

        if not valid_tasks:
            return slices_added

        slices_added_per_pid = 0
        failed_searches      = 0
        _MAX_TRIES           = 5   # random valid-slice candidates per outer iteration

        while (slices_added_per_pid < 150
               and slices_added      < max_data_points
               and failed_searches   < 100):

            primary_task  = random.choice(valid_tasks)
            candidates    = valid_index[primary_task]

            # Build a frozenset for O(1) reference-frame lookup (built once per
            # outer-loop iteration, not per candidate try).
            ref_valid_set = frozenset(candidates.tolist())

            # Sample up to _MAX_TRIES slices that are already known to have foreground.
            n_tries = min(len(candidates), _MAX_TRIES)
            chosen  = np.random.choice(candidates, size=n_tries, replace=False)

            found = False
            for i in chosen.tolist():
                r = self._sample_offset(i, offset, y_shape)
                if r is None:
                    continue

                # Pre-check: reference frame (i+r) must also be a valid slice for
                # this task.  Avoids the expensive _create_single_datapoint call
                # when the reference mask would be empty.
                if (i + r) not in ref_valid_set:
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
                    break

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
        self._universeg_cache.clear()
        self._slice_index_cache.clear()

        x_new, y_new, prompt, offset_list, m_new = [], [], [], [], []
        slices_added = 0

        while slices_added < max_data_points:
            random.shuffle(self.dataloader.current_ids)

            for pid in self.dataloader.current_ids:
                current_dict = self.dataloader.dataset[pid]
                modality_str = str(current_dict.get('modality', 'UNKNOWN')).lower()
                is_mri = 0.0 if 'ct' in modality_str else 1.0

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
                        is_mri, slices_added, max_data_points, pid=pid
                    )

                if slices_added >= max_data_points:
                    break

        x_new, y_new, prompt, offset_list, m_new = self._randomize(
            x_new, y_new, prompt, offset_list, m_new
        )
        print(f'It took {time.time() - start:.0f} seconds')
        return x_new, y_new, prompt, offset_list, m_new

    @staticmethod
    def _make_object_array(lst):
        """Safely create a 1-D object array from a list of variable-shape arrays.

        ``np.array(lst, dtype=object)`` can raise a broadcast error when all
        arrays share one common dimension (e.g. all (222, ?, 1)) because NumPy
        infers a higher-dimensional layout instead of a flat object array.
        Pre-allocating and assigning element-by-element is always safe.
        """
        arr = np.empty(len(lst), dtype=object)
        for i, a in enumerate(lst):
            arr[i] = a
        return arr

    def _to_numpy_arrays(self, x_lst, y_lst, p_lst, m_lst):
        """Stack lists of arrays. If shapes vary (native mode), creates object arrays."""
        try:
            return (
                np.stack(x_lst).astype(np.float32),
                np.stack(y_lst).astype(np.float32),
                np.stack(p_lst).astype(np.float32),
                np.array(m_lst, dtype=np.float32),
            )
        except ValueError:
            return (
                self._make_object_array(x_lst),
                self._make_object_array(y_lst),
                self._make_object_array(p_lst),
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

    # ---- Single-task generation (for UniverSeg-style eval data) --------- #

    def get_data_points_from_one_task_numpy(self, max_data_points=116, offset=5,
                                            dimensions=None,
                                            extraction_mode='native',
                                            return_task=False):
        """
        Generate data points that ALL belong to the SAME segmentation task
        (anatomical structure).  Unlike the multi-task collectors, every sample
        in the returned arrays shares one segment class, making this suitable
        for building UniverSeg / few-shot evaluation datasets where the support
        set and query set must cover the same structure.

        Algorithm
        ---------
        1. Pick a random dimension and a random task globally.
        2. Iterate over all patients; for each one that contains the task,
           harvest (x, y, prompt, x_u) tuples using _create_single_datapoint.
        3. If >= 32 samples collected → shuffle and return.
        4. If fewer than 32 → discard and repeat from step 1 with a new task.

        Parameters
        ----------
        extraction_mode : 'native' (recommended), 'crop' or 'fullslice'
            'native'    → keeps exact native slice shape matching the source volume
                          (_extract_native_2d). Returns variable-sized arrays.
            'crop'      → label-guided 128–256 px patch crop (_extract_patch_2d).
                          In-plane dimensions must be >= 128; smaller volumes
                          are skipped automatically.
            'fullslice' → maximum centered square (min(H, W) × min(H, W))
                          resized to 128×128 (_extract_fullslice_2d).  Works
                          for any in-plane size; nothing is skipped.

        Returns
        -------
        x_np   : (N, 128, 128, 1)  float32  — z-score images    [-5, 5]
        y_np   : (N, 128, 128, 1)  float32  — binary labels
        p_np   : (N, 128, 128, 2)  float32  — prompts [ref_img | ref_label]
        x_u_np : (N, 128, 128, 1)  float32  — UniverSeg images  [0, 1]
        m_np   : (N,)              float32  — 0.0 = CT, 1.0 = MRI
        offsets: list[int]
        """
        if dimensions is None:
            dimensions = ['x', 'y', 'z']

        self.dataloader.current_ids = self.dataloader.train_ids
        # NOTE: caches are intentionally NOT cleared here.
        # The underlying data is static across all dataset generation calls,
        # so norm, universeg, and slice-index results remain valid.

        start = time.time()
        print("Creating new Data Points ...")

        while True:
            x_new, y_new, prompt, offset_list, m_new, xu_new = [], [], [], [], [], []
            slices_added = 0
            step_counter = 0

            # --- 1. Pick a random dimension and task ---------------------
            d = random.choice(dimensions)
            task = None

            random.shuffle(self.dataloader.current_ids)
            for pid in self.dataloader.current_ids:
                current_dict = self.dataloader.dataset[pid]
                y_raw = current_dict['segmentations']
                if isinstance(y_raw, list) and len(y_raw) == 1:
                    y_raw = np.asarray(y_raw[0])

                if isinstance(y_raw, list):
                    # Multi-channel: task = 1-based channel index
                    task = random.randint(1, len(y_raw))
                    break
                else:
                    y_arr = np.asarray(y_raw)
                    valid = [int(v) for v in np.unique(y_arr) if v != 0]
                    if valid:
                        task = random.choice(valid)
                        break

            if task is None:
                raise RuntimeError(
                    "No valid segmentation tasks found in the dataset."
                )
            print(f'Current task (Global): {task}')

            # --- 2. Collect slices for this task across all patients -----
            random.shuffle(self.dataloader.current_ids)
            for pid in self.dataloader.current_ids:
                if slices_added >= max_data_points:
                    break
                if step_counter >= max_data_points * 20:
                    break

                current_dict = self.dataloader.dataset[pid]
                modality_str = str(current_dict.get('modality', 'UNKNOWN')).lower()
                is_mri = 0.0 if 'ct' in modality_str else 1.0

                y_raw = current_dict['segmentations']
                if isinstance(y_raw, list) and len(y_raw) == 1:
                    y_raw = np.asarray(y_raw[0])

                dim_idx = 'xyz'.index(d)

                if isinstance(y_raw, list):
                    vol_shape = np.asarray(y_raw[0]).shape
                else:
                    vol_shape = np.asarray(y_raw).shape

                # 'crop' mode requires the patch to fit inside the slice.
                # 'fullslice' resizes whatever it gets, so any size is valid.
                slice_dims = [s for j, s in enumerate(vol_shape) if j != dim_idx]
                if extraction_mode == 'crop' and min(slice_dims) < max(self.height, self.width):
                    continue

                # Use pre-computed valid-slice index to skip patients / slices
                # that don't contain the task — the cache already tells us this.
                valid_index   = self._get_slice_index(y_raw, pid, dim_idx)
                candidate_arr = valid_index.get(task, np.array([], dtype=np.int64))
                if candidate_arr.size == 0:
                    continue  # patient has no valid slices for this task in this dimension

                x, y  = self._prepare_volume(current_dict, pid=pid)
                x_u   = self._prepare_universeg_volume(current_dict, pid=pid)
                if isinstance(y, list) and len(y) == 1:
                    y = np.asarray(y[0])

                y_shape = vol_shape[dim_idx]

                slice_indices = candidate_arr.copy()
                np.random.shuffle(slice_indices)

                for i in slice_indices:
                    i = int(i)
                    if slices_added >= max_data_points:
                        break
                    step_counter += 1
                    if step_counter >= max_data_points * 20:
                        break

                    r = self._sample_offset(i, offset, y_shape)
                    if r is None:
                        continue

                    result = self._create_single_datapoint(
                        x, y, i, r, d,
                        max_number_labels=1,
                        primary_task=task,
                        x_u=x_u,
                        extraction_mode=extraction_mode,
                    )
                    if result is None:
                        continue

                    x2d, y2d, p, xu_2d = result
                    x_new.append(x2d)
                    y_new.append(y2d)
                    prompt.append(p)
                    xu_new.append(xu_2d)
                    offset_list.append(r)
                    m_new.append(is_mri)
                    slices_added += 1

            # --- 3. Decide: proceed or retry with a new task -------------
            if slices_added >= 32:
                if slices_added < max_data_points:
                    print(
                        f"Task exhausted, but collected {slices_added} "
                        f"data points. Proceeding with dataset."
                    )
                break
            else:
                print("Changed the task, because it was exhausted across all patients.")
                # Loop to pick a new task

        # Shuffle all six lists in unison
        combined = list(zip(x_new, y_new, prompt, offset_list, m_new, xu_new))
        random.shuffle(combined)
        if combined:
            x_new, y_new, prompt, offset_list, m_new, xu_new = zip(*combined)
            x_new, y_new, prompt = list(x_new), list(y_new), list(prompt)
            offset_list, m_new, xu_new = list(offset_list), list(m_new), list(xu_new)

        print(f'It took {time.time() - start:.0f} seconds')

        x_np, y_np, p_np, m_np = self._to_numpy_arrays(x_new, y_new, prompt, m_new)
        
        try:
            x_u_np = np.stack(xu_new).astype(np.float32)
        except ValueError:
            x_u_np = self._make_object_array(xu_new)
        
        if return_task:
            return x_np, y_np, p_np, x_u_np, m_np, offset_list, task
        return x_np, y_np, p_np, x_u_np, m_np, offset_list

    def get_data_points_from_one_task(self, max_data_points=116, offset=5,
                                      dimensions=None):
        """
        tf.data.Dataset wrapper around get_data_points_from_one_task_numpy.
        Yields (x, y, prompt, modality) tuples — kept for backward compatibility.
        Prefer get_data_points_from_one_task_numpy() for direct numpy usage.

        Note: x_u_np (UniverSeg normalisation) is available from
        get_data_points_from_one_task_numpy() but is not included in the
        tf.data.Dataset as it is only needed for evaluation bundle generation.
        """
        import tensorflow as tf
        x_np, y_np, p_np, x_u_np, m_np, offset_list = \
            self.get_data_points_from_one_task_numpy(
                max_data_points, offset, dimensions
            )
        ds = tf.data.Dataset.from_tensor_slices((x_np, y_np, p_np, m_np))
        return ds, offset_list