import random
import math
import time

import numpy as np
import tensorflow as tf

from utils.preprocessing import universal_normalization

class DataGenerator():
    """
    __init__ directly loads data with the DataLoader Object. (To create prompt/datapoints call get_data_points() or get_val_data_points())
    Parameters:
    dataloader: Strategy Pattern. DL must store the data in a dict that looks like dict[id] = {'image': volume, 'segmentations': volume_of_labels}. 
        'segmentations' can be a single segmentation where each int displays a different region or several binary segmentation masks.
    img_height: ...
    img_width: ...
    minimum_pixel: Number of pixel that needs to be in each segmentation (y and y+i) in order to be used for a new DP. 
    """
    # Need to add relative minimal cropping volume size because head data is smaller
    
    def __init__(self, dataloader, img_height=128, img_width=128, minimum_pixel=5):
        
        # Dataset and further information is stored in the concrete DataLoader object
        self.dataloader = dataloader
        
        self.height = img_height
        self.width = img_width
        self.minimum_pixel = minimum_pixel

        # Cache normalized volumes for the duration of one data-generation call.
        # Key: patient id string  Value: (x_normalized, segs)
        # Cleared at the start of every get_data_points / get_val_data_points call
        # so stale data never leaks between training epochs.
        self._norm_cache: dict = {}
    
    # -------------------------- Helpers --------------------------

    def _get_2d_data(self, img, slice_idx, axis):
        try:
            if axis == 'x':
                return img[slice_idx, :, :]
            elif axis == 'y':
                return img[:, slice_idx, :]
            else:  # 'z'
                return img[:, :, slice_idx]
        except Exception as e:
            print(f"Error slicing {axis} axis at index {slice_idx} with shape {img.shape}")
            raise e
    

    def _randomnizer(self, x_new, y_new, prompt, offset_list, m_new, dimensions):
        #Randomize 3 axis
        if len(dimensions) > 1:
            zipped_lists = list(zip(x_new, y_new, prompt, offset_list, m_new))
            random.shuffle(zipped_lists)
            x_new, y_new, prompt, offset_list, m_new = zip(*zipped_lists)
            x_new, y_new, prompt, offset_list, m_new = list(x_new), list(y_new), list(prompt), list(offset_list), list(m_new)
        return x_new, y_new, prompt, offset_list, m_new
    
    def _extract_patch_2d(self, x_2d, total_label, total_label_r, x_2d_r):
        """
        Extract a **label-guided** 128×128 patch from a 2-D slice pair.

        The crop window is constrained so it is guaranteed to overlap with the
        bounding box of `total_label`. This prevents the pathological case where
        a random window lands in an empty background region and the resulting
        data point contains no segmentation mask.

        Crop selection logic
        --------------------
        Given the label bounding box [min_h..max_h] × [min_w..max_w] in the
        padded slice, the valid top-left origin (start_h, start_w) must satisfy:

            max_h - patch + 1  <=  start_h  <=  min_h
            max_w - patch + 1  <=  start_w  <=  min_w

        which ensures the patch includes at least one labeled pixel. The origin
        is chosen uniformly at random within this range, so we still get
        variety in exactly how much of the structure is visible.

        If `total_label` is entirely empty (should not happen after
        _select_valid_labels, but handled defensively) we fall back to a
        fully random crop.

        Because volumes are stored at 1 mm isotropic resolution we never
        resize – a 128-voxel window always represents 12.8 cm.

        Args:
            x_2d, x_2d_r        : 2-D array/tensor of shape (H, W).
            total_label         : 2-D binary mask of shape (H, W) – primary slice.
            total_label_r       : 2-D binary mask of shape (H, W) – reference slice.

        Returns:
            x_2d, x_2d_r, total_label, total_label_r – all shape (1, 128, 128, 1).
        """
        patch_size = self.height  # 128

        def pad_if_needed(t, target, pad_val=0.0):
            """Symmetrically zero-pad tensor `t` (H, W) so both dims >= target."""
            h, w = t.shape[0], t.shape[1]
            pad_h = max(0, target - h)
            pad_w = max(0, target - w)
            pad_top    = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left   = pad_w // 2
            pad_right  = pad_w - pad_left
            return tf.pad(t, [[pad_top, pad_bottom], [pad_left, pad_right]], constant_values=pad_val)

        x_2d          = pad_if_needed(x_2d,          patch_size, pad_val=-5.0)
        x_2d_r        = pad_if_needed(x_2d_r,        patch_size, pad_val=-5.0)
        total_label   = pad_if_needed(total_label,   patch_size, pad_val=0.0)
        total_label_r = pad_if_needed(total_label_r, patch_size, pad_val=0.0)

        h, w = x_2d.shape[0], x_2d.shape[1]

        # ---- Label-guided crop position ----------------------------------------
        # Convert to numpy (we are always in eager mode here) so we can use
        # np.argwhere for the bounding-box computation.
        label_np = total_label.numpy() if hasattr(total_label, 'numpy') else np.asarray(total_label)
        nonzero  = np.argwhere(label_np > 0)        # shape [N, 2]

        if len(nonzero) > 0:
            min_h_l, min_w_l = nonzero.min(axis=0)
            max_h_l, max_w_l = nonzero.max(axis=0)

            # Valid start-row range so the patch overlaps the label bbox.
            lo_h = int(max(0,            max_h_l - patch_size + 1))
            hi_h = int(min(h - patch_size, min_h_l))
            hi_h = max(lo_h, hi_h)   # safety-clamp in case bbox > patch_size

            lo_w = int(max(0,            max_w_l - patch_size + 1))
            hi_w = int(min(w - patch_size, min_w_l))
            hi_w = max(lo_w, hi_w)

            start_h = random.randint(lo_h, hi_h)
            start_w = random.randint(lo_w, hi_w)
        else:
            # Fallback – label is empty (defensive; should not happen here).
            start_h = random.randint(0, max(0, h - patch_size))
            start_w = random.randint(0, max(0, w - patch_size))
        # ------------------------------------------------------------------------

        def crop(t):
            return t[start_h : start_h + patch_size, start_w : start_w + patch_size]

        x_2d          = crop(x_2d)
        x_2d_r        = crop(x_2d_r)
        total_label   = crop(total_label)
        total_label_r = crop(total_label_r)

        # Add channel dim → (128, 128, 1)
        x_2d          = tf.cast(x_2d[..., tf.newaxis], tf.float32)
        x_2d_r        = tf.cast(x_2d_r[..., tf.newaxis], tf.float32)
        total_label   = tf.cast(total_label[..., tf.newaxis], tf.float32)
        total_label_r = tf.cast(total_label_r[..., tf.newaxis], tf.float32)

        return x_2d, x_2d_r, total_label, total_label_r

    def _prepare_volume(self, current_dict, pid=None):
        """
        Normalize the image volume from the dataset dict.

        Volumes are already at 1 mm isotropic resolution (resampled during NPZ
        generation), so no spatial resampling is performed here.

        Parameters
        ----------
        current_dict : dict
            Single-patient entry from the dataloader.
        pid : str, optional
            Patient ID used as cache key. When provided the normalized volume is
            stored in ``self._norm_cache`` and reused on subsequent visits within
            the same data-generation call, avoiding repeated 3-D TF operations.
        """
        if pid is not None and pid in self._norm_cache:
            return self._norm_cache[pid]

        segs     = current_dict['segmentations']
        x        = current_dict['image']
        modality = current_dict.get('modality', 'UNKNOWN')

        x = universal_normalization(x, modality=modality)

        result = (x, segs)
        if pid is not None:
            self._norm_cache[pid] = result
        return result


    def _process_dimension(self, x, y, d, offset, max_number_labels, x_new, y_new, prompt, offset_list, m_new, modality_flag, slices_added, max_data_points):
        # Skip dimension if the resulting 2D slice is strictly smaller than the patch size.
        # This prevents the generator from padding thin slices which results in "squashed line" artifacts.
        dim_idx = 'xyz'.index(d)
        slice_dims = [s for i, s in enumerate(x.shape) if i != dim_idx]
        if min(slice_dims) < max(self.height, self.width):
            return slices_added

        slices_added_per_pid = 0
        
        # 1. Determine fully available Tasks globally for this patient
        if isinstance(y, list):
            y_shape = y[0].shape['xyz'.index(d)]   
            valid_tasks = list(range(1, len(y) + 1))
        else:
            y_shape = y.shape['xyz'.index(d)]
            y_flat = tf.reshape(y, [-1])
            valid_tasks_tensor, _ = tf.unique(y_flat)
            valid_tasks = valid_tasks_tensor.numpy().tolist()
            if 0 in valid_tasks:
                valid_tasks.remove(0)
                
        # If no identifiable tasks exist at all in the scan, skip it.
        if not valid_tasks:
            return slices_added

        failed_searches = 0
        
        # 2. Iterate dynamically until we satisfy requested generated points
        while slices_added_per_pid < 150 and slices_added < max_data_points and failed_searches < 100:
            
            # Step A: Pick primary task uniformly across all tasks
            primary_task = random.choice(valid_tasks)
            
            # Step B: Pick slices dynamically until we find one containing it
            slice_indices = list(range(y_shape))
            random.shuffle(slice_indices)

            found_slice = False
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
                    slices_added += 1
                    slices_added_per_pid += 1
                    found_slice = True
                    failed_searches = 0
                    break # Break slice hunt to loop & pick a new randomized task

            if not found_slice:
                failed_searches += 1

        return slices_added
    
    def _sample_offset(self, i, offset, y_shape):
        possible = list(range(-offset, 0)) + list(range(1, offset + 1))
        r = random.choice(possible)

        if (i + r < 0) or (i + r >= y_shape):
            return None
        return r
    
    def _create_single_datapoint(self, x, y, i, r, d, max_number_labels, primary_task=None):
        # multiple segmentations = list case
        if isinstance(y, list):
            y_2d = [self._get_2d_data(seg, i, d) for seg in y]
            y_2d_r = [self._get_2d_data(seg, i+r, d) for seg in y]
        else:
            y_2d = self._get_2d_data(y, i, d)
            y_2d_r = self._get_2d_data(y, i+r, d)


        labels, labels_r = self._select_valid_labels(y_2d, y_2d_r, max_number_labels, primary_task)
        if labels is None:
            return None

        total_label = self._merge_labels(labels)
        total_label_r = self._merge_labels(labels_r)

        x_2d = self._get_2d_data(x, i, d)
        x_2d_r = self._get_2d_data(x, i + r, d)

        x_2d, x_2d_r, total_label, total_label_r = self._extract_patch_2d(
            x_2d, total_label, total_label_r, x_2d_r
        )

        # Post-crop validation -----------------------------------------------
        # _extract_patch_2d guarantees the PRIMARY label (total_label) overlaps
        # the crop window, but total_label_r (reference slice at i+r) is cropped
        # at the same position without guidance.  If the anatomy shifted between
        # slices (e.g. offset=5 → 5mm), the reference label may land outside
        # the 128×128 window → invisible segmentation in the prompt.
        # Discard the datapoint if either mask is too sparse after cropping.
        if (tf.math.count_nonzero(total_label)   < self.minimum_pixel or
                tf.math.count_nonzero(total_label_r) < self.minimum_pixel):
            return None
        # --------------------------------------------------------------------

        p = tf.concat([x_2d_r, total_label_r], axis=-1)
        return x_2d, total_label, p

    
    def _select_valid_labels(self, y_2d, y_2d_r, max_number_labels, primary_task):
        """
        Modified version:
        Case 1: Multi-segmentation list. Binarizes inputs (0 vs >0). 
        Case 2: Single segmentation volume. Filters unique INTs, removes 0.
        
        Anchors on primary_task: if it is missing, returns None. 
        If present, fetches secondary tasks organically up to count.
        """
        with tf.device('/CPU:0'):
            count = random.randint(1, max_number_labels)
            labels_out, labels_r_out = [], []

            # Determine if we are truly in "Multi-Channel List" mode or "Single Volume" mode
            is_multi_channel_list = False

            if isinstance(y_2d, list):
                if len(y_2d) > 1:
                    is_multi_channel_list = True
                elif len(y_2d) == 1:
                    # Unwrap the list of length 1 to treat it as a single multi-label volume
                    y_2d = y_2d[0]
                    y_2d_r = y_2d_r[0]
                    is_multi_channel_list = False
                else:
                    return None, None

            # ---------------- CASE 1: List of Multiple Channels ----------------
            if is_multi_channel_list:
                
                # Check for primary_task (1-indexed map to idx)
                if primary_task is not None:
                    idx = primary_task - 1
                    s1 = tf.where(y_2d[idx] > 0, 1, 0)
                    s2 = tf.where(y_2d_r[idx] > 0, 1, 0)
                    
                    if tf.math.count_nonzero(s1) < self.minimum_pixel or tf.math.count_nonzero(s2) < self.minimum_pixel:
                        return None, None
                        
                    labels_out.append(s1)
                    labels_r_out.append(s2)
                    
                    if count > 1:
                        # Extract secondary tasks naturally occurring in this slice
                        other_indices = []
                        for other_idx, (seg, seg_r) in enumerate(zip(y_2d, y_2d_r)):
                            if other_idx == idx: continue
                            s1_o = tf.where(seg > 0, 1, 0)
                            s2_o = tf.where(seg_r > 0, 1, 0)
                            if tf.math.count_nonzero(s1_o) >= self.minimum_pixel and tf.math.count_nonzero(s2_o) >= self.minimum_pixel:
                                other_indices.append(other_idx)
                        random.shuffle(other_indices)
                        
                        for c_idx in other_indices[:count - 1]:
                            l = tf.where(y_2d[c_idx] > 0, 1, 0)
                            l_r = tf.where(y_2d_r[c_idx] > 0, 1, 0)
                            labels_out.append(l)
                            labels_r_out.append(l_r)

            # ---------------- CASE 2: Single Volume (Multi-Label Integers) ----------------
            else: 
                if primary_task is not None:
                    # Create mask for primary task
                    label = tf.where(y_2d == primary_task, 1, 0)
                    label_r = tf.where(y_2d_r == primary_task, 1, 0)
                    if tf.math.count_nonzero(label) < self.minimum_pixel or tf.math.count_nonzero(label_r) < self.minimum_pixel:
                        return None, None
                        
                    labels_out.append(label)
                    labels_r_out.append(label_r)
                    
                    if count > 1:
                        y1_flat = tf.reshape(y_2d, [-1])
                        y2_flat = tf.reshape(y_2d_r, [-1])

                        valid_1, _ = tf.unique(y1_flat)
                        valid_2, _ = tf.unique(y2_flat)

                        set_1 = set(valid_1.numpy())
                        set_2 = set(valid_2.numpy())

                        set_1.discard(0) 
                        set_2.discard(0)
                        set_1.discard(primary_task)
                        set_2.discard(primary_task)

                        candidates = list(set_2.intersection(set_1))
                        random.shuffle(candidates)

                        for label_val in candidates:
                            l = tf.where(y_2d == label_val, 1, 0)
                            l_r = tf.where(y_2d_r == label_val, 1, 0)

                            if (tf.math.count_nonzero(l) < self.minimum_pixel or 
                                tf.math.count_nonzero(l_r) < self.minimum_pixel):
                                continue

                            labels_out.append(l)
                            labels_r_out.append(l_r)

                            if len(labels_out) == count:
                                break

            # ---------------- FINAL RETURN ----------------
            if len(labels_out) == 0:
                return None, None

            return labels_out, labels_r_out


    def _merge_labels(self, labels):
        result = labels[0]
        for l in labels[1:]:
            result = result + l
        return result
    

    # -------------------------- Generators --------------------------
    
    def _get_data_points(self, max_data_points, offset, max_number_labels, dimensions):
        start = time.time()
        print("Creating new Data Points ...")

        # Clear the normalization cache so we normalize each patient exactly once
        # per data-generation call instead of on every loop iteration.
        self._norm_cache.clear()

        offset_list = []
        x_new, y_new, prompt, m_new = [], [], [], []
        slices_added = 0

        while slices_added < max_data_points:

            random.shuffle(self.dataloader.current_ids)

            for id in self.dataloader.current_ids:

                current_dict = self.dataloader.dataset[id]
                modality_str = current_dict.get('modality', 'UNKNOWN')
                is_mri = 0.0 if modality_str == 'CT' else 1.0

                x, y = self._prepare_volume(current_dict, pid=id)
                
                # Unwrap single-element list to treat as a single multi-label volume
                if isinstance(y, list) and len(y) == 1:
                    y = y[0]
                    
                random.shuffle(dimensions)

                for d in dimensions:

                    if slices_added >= max_data_points:
                        break

                    slices_added = self._process_dimension(
                        x, y, d, offset, max_number_labels,
                        x_new, y_new, prompt, offset_list, m_new, is_mri, slices_added, 
                        max_data_points)

                    if slices_added >= max_data_points:
                        break

        # Finalize dataset
        x_new, y_new, prompt, offset_list, m_new = self._randomnizer(x_new, y_new, prompt, offset_list, m_new, dimensions)
        ds = tf.data.Dataset.from_tensor_slices((
            tf.stack(x_new), tf.stack(y_new), tf.stack(prompt), tf.cast(m_new, tf.float32)
        ))

        print(f'It took {time.time() - start:.0f} seconds')
        return ds, offset_list

    
    def _get_data_points_from_one_task(self, max_data_points, offset, dimensions):
        
        start = time.time()
        print("Creating new Data Points ...")

        offset_list = []
        x_new, y_new, prompt, m_new = [], [], [], []
        slices_added = 0
        task = 0

        d = random.choice(dimensions)

        while slices_added < max_data_points:

            if task == 0:
                # Re-pick building blocks
                d = random.choice(dimensions)
                # Pick a valid task globally from a random patient
                random.shuffle(self.dataloader.current_ids)
                for id in self.dataloader.current_ids:
                    current_dict = self.dataloader.dataset[id]
                    y = current_dict['segmentations']
                    if isinstance(y, list) and len(y) == 1: y = y[0]

                    if isinstance(y, list):
                        task = random.randint(1, len(y))
                        break
                    else:
                        y_flat = tf.reshape(y, [-1])
                        valid_tasks, _ = tf.unique(y_flat)
                        valid_tasks = valid_tasks.numpy().tolist()
                        if 0 in valid_tasks: valid_tasks.remove(0)
                        if len(valid_tasks) > 0:
                            task = random.choice(valid_tasks)
                            break
                print(f'Current task (Global): {task}')
                step_counter = 0

            # Iterate over ALL patients for this specific task
            random.shuffle(self.dataloader.current_ids)
            for id in self.dataloader.current_ids:
                if slices_added >= max_data_points:
                    break
                    
                if step_counter >= max_data_points * 20:
                    break # Give up if task is extremely sparse

                current_dict = self.dataloader.dataset[id]
                modality_str = current_dict.get('modality', 'UNKNOWN')
                is_mri = 0.0 if modality_str == 'CT' else 1.0

                x, y = self._prepare_volume(current_dict, pid=id)
                if isinstance(y, list) and len(y) == 1:
                    y = y[0]

                # Pre-check if dimension is too small and would require 0-padding
                dim_idx = 'xyz'.index(d)
                slice_dims = [s for j, s in enumerate(x.shape) if j != dim_idx]
                if min(slice_dims) < max(self.height, self.width):
                    continue

                # Pre-check if patient actually has the desired task
                if isinstance(y, list):
                    if tf.math.count_nonzero(y[task-1]) == 0:
                        continue
                    y_shape = y[task-1].shape['xyz'.index(d)]
                else:
                    y_flat = tf.reshape(y, [-1])
                    valid_tasks, _ = tf.unique(y_flat)
                    valid_tasks = valid_tasks.numpy().tolist()
                    if task not in valid_tasks:
                        continue
                    y_shape = y.shape['xyz'.index(d)]

                iter_2d = tf.range(y_shape)
                tf.random.shuffle(iter_2d)

                for i in iter_2d:
                    if slices_added >= max_data_points:
                        break
                        
                    step_counter += 1

                    r = self._sample_offset(i, offset, y_shape)
                    if r is None:
                        continue

                    if isinstance(y, list):
                        y_2d = self._get_2d_data(y[task-1], i, d)
                        y_2d_r = self._get_2d_data(y[task-1], i + r, d)
                    else:
                        y_2d = self._get_2d_data(y, i, d)
                        y_2d_r = self._get_2d_data(y, i + r, d)

                    # Task label verification check
                    if isinstance(y, list):
                         label = tf.where(y_2d > 0, 1, 0)
                         label_r = tf.where(y_2d_r > 0, 1, 0)
                    else:
                         label = tf.where(y_2d == task, 1, 0)
                         label_r = tf.where(y_2d_r == task, 1, 0)

                    if (tf.math.count_nonzero(label) < self.minimum_pixel or
                        tf.math.count_nonzero(label_r) < self.minimum_pixel):
                        continue

                    # Process and resize
                    total_label = label
                    total_label_r = label_r
                    x_2d = self._get_2d_data(x, i, d)
                    x_2d_r = self._get_2d_data(x, i + r, d)

                    x_2d, x_2d_r, total_label, total_label_r = self._extract_patch_2d(
                        x_2d, total_label, total_label_r, x_2d_r
                    )
                    p = tf.concat([x_2d_r, total_label_r], axis=-1)

                    x_new.append(x_2d)
                    y_new.append(total_label)
                    prompt.append(p)
                    offset_list.append(r)
                    m_new.append(is_mri)
                    slices_added += 1

            if slices_added < max_data_points:
                if slices_added >= 32:
                    print(f"Task exhausted, but collected {slices_added} data points. Proceeding with dataset.")
                    break

                print("Changed the task, because it was exhausted across all patients.")
                task = 0
                slices_added = 0
                offset_list.clear() # Clearing correctly since it's a list
                m_new.clear()
                x_new.clear()
                y_new.clear()
                prompt.clear()

        # Shuffle / stack
        x_new, y_new, prompt, offset_list, m_new = self._randomnizer(
            x_new, y_new, prompt, offset_list, m_new, dimensions
        )
        ds = tf.data.Dataset.from_tensor_slices((
            tf.stack(x_new), tf.stack(y_new), tf.stack(prompt), tf.cast(m_new, tf.float32)
        ))

        print(f'It took {(time.time() - start):.0f} seconds')
        return ds, offset_list
    
    
    # -------------------------- Public Getters --------------------------

    def get_data_points(self, max_data_points=3500, offset=5, max_number_labels=1, dimensions=['x','y','z']):
        """
        Generate training data points.
        Each element of the returned dataset is a tuple (x, y, prompt), all
        shape (1, 128, 128, 1) or (1, 128, 128, 2) for the prompt.
        """
        self.dataloader.current_ids = self.dataloader.train_ids
        ds, offsets = self._get_data_points(max_data_points, offset, max_number_labels, dimensions)
        return ds, offsets

    def get_val_data_points(self, max_data_points=3500, offset=5, max_number_labels=1, dimensions=['x','y','z']):
        self.dataloader.current_ids = self.dataloader.validation_ids
        ds, offsets = self._get_data_points(max_data_points, offset, max_number_labels, dimensions)
        return ds, offsets

    def get_data_points_from_one_task(self, max_data_points=3500, offset=5, dimensions=['x','y','z']):
        """Generate data points from a single randomly selected task.
        Returns: tf.Dataset. One Task from one random patient. Each element of ds = (x, y, p)
        """
        self.dataloader.current_ids = self.dataloader.train_ids
        ds, offsets = self._get_data_points_from_one_task(max_data_points, offset, dimensions)
        return ds, offsets