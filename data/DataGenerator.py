import random
import math
import time

import tensorflow as tf

from utils.preprocessing import min_max_norm


class DataGenerator():
    """
    __init__ directly loads data with the DataLoader Object. (To create prompt/datapoints call get_data_points() or get_val_data_points())
    Parameters:
    dataloader: Strategy Pattern. DL must store the data in a dict that looks like dict[id] = {'image': volume, 'segmentations': volume_of_labels}. 
        'segmentations' can be a single segmentation where each int displays a different region or several binary segmentation masks.
    img_height: ...
    img_width: ...
    minimum_pixel: Number of pixel that need to be in both segmentations (y and y+i) in order to be used for a new DP. 
    """
    # Need to add relative minimal cropping volume size because head data is smaller
    
    def __init__(self, dataloader, img_height=128, img_width=128, minimum_pixel=25):
        
        # Dataset and further information is stored in the concrete DataLoader object
        self.dataloader = dataloader
        
        self.height = img_height
        self.width = img_width
        self.minimum_pixel = minimum_pixel
    
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
    
    def _cast_norm_resize(self, x_2d, x_2d_r, total_label, total_label_plus_r):
        # Resize and concat only works with 3D (h,w,channels)
        x_2d = tf.expand_dims(x_2d, axis=-1)
        x_2d_r = tf.expand_dims(x_2d_r, axis=-1)
        total_label_plus_r = tf.expand_dims(total_label_plus_r, axis=-1)
        total_label = tf.expand_dims(total_label, axis=-1)

        # Normalization between 0 and 1. Segm. and Prompt segm. are already binary
        x_2d = min_max_norm(x_2d)
        x_2d_r = min_max_norm(x_2d_r)
        
        x_2d = tf.cast(x_2d, tf.float32)
        x_2d_r = tf.cast(x_2d_r, tf.float32)
        total_label = tf.cast(total_label, tf.float32)
        total_label_plus_r = tf.cast(total_label_plus_r, tf.float32)

        total_label = tf.image.resize(total_label, [self.height, self.width], method='nearest') # important to use nearest for segmentations
        total_label_plus_r = tf.image.resize(total_label_plus_r, [self.height, self.width], method='nearest')
        x_2d = tf.image.resize(x_2d, [self.height, self.width])
        x_2d_r = tf.image.resize(x_2d_r, [self.height, self.width])

        return x_2d, x_2d_r, total_label, total_label_plus_r
    
    def _randomnizer(self, x_new, y_new, prompt, offset_list, dimensions):
        #Randomize 3 axis
        if len(dimensions) > 1:
            zipped_lists = list(zip(x_new, y_new, prompt, offset_list))
            random.shuffle(zipped_lists)
            x_new, y_new, prompt, offset_list = zip(*zipped_lists)
            x_new, y_new, prompt, offset_list = list(x_new), list(y_new), list(prompt), list(offset_list)
        return x_new, y_new, prompt, offset_list
    
    def _random_3d_crop(self, volume, mask, min_crop_size):
        """
        Perform the same random 3D crop on two 3D NumPy arrays.

        Parameters:
        - volume: Input 3D array (e.g., shape [D, H, W])
        - mask: Mask 3D array (e.g., shape [D, H, W])
        - crop_size (tuple): Desired crop size (cd, ch, cw)

        Returns:
        - Cropped 3D arrays of shape (cd, ch, cw)
        - Tuple: Contains information how the image was cropped
        """
        d, h, w = volume.shape

        min_cd, min_ch, min_cw = min_crop_size

        # Choose size of new volume randomly between minimal size and shape of the original volume as maximum
        cd = tf.random.uniform(shape=[], minval=min_cd, maxval=d, dtype=tf.int32)
        ch = tf.random.uniform(shape=[], minval=min_ch, maxval=h, dtype=tf.int32)
        cw = tf.random.uniform(shape=[], minval=min_cw, maxval=w, dtype=tf.int32)

        if cd > d or ch > h or cw > w:
            raise ValueError("Crop size must be smaller than the volume size in all dimensions.")

        start_d = tf.random.uniform(shape=[], minval=0, maxval=d - cd + 1, dtype=tf.int32)
        start_h = tf.random.uniform(shape=[], minval=0, maxval=h - ch + 1, dtype=tf.int32)
        start_w = tf.random.uniform(shape=[], minval=0, maxval=w - cw + 1, dtype=tf.int32)

        crop = volume[start_d:start_d+cd, start_h:start_h+ch, start_w:start_w+cw]
        crop_mask = mask[start_d:start_d+cd, start_h:start_h+ch, start_w:start_w+cw]

        return crop, crop_mask, (start_d, start_d+cd), (start_h, start_h+ch), (start_w, start_w+cw)
    
    def _prepare_volume(self, current_dict, cropping, min_crop_size):
        """
        Prepare image and segmentation volumes.
        min_crop_size is now a percentage (e.g., 0.5 → 50% of each dimension).
        """
        segs = current_dict['segmentations']
        x = current_dict['image']

        # Determine dynamic min_crop_size: percentage of volume
        pct = float(min_crop_size)
        if pct <= 0 or pct > 1:
            raise ValueError("min_crop_size must be in the range (0,1].")
        min_cd = max(1, int(x.shape[0] * pct))
        min_ch = max(1, int(x.shape[1] * pct))
        min_cw = max(1, int(x.shape[2] * pct))
        min_crop_size = (min_cd, min_ch, min_cw)

        # Multi-segmentation case
        if isinstance(segs, list):
            if cropping:
                # Crop using first segmentation to compute coordinates
                x_crop, _, crop_d, crop_h, crop_w = self._random_3d_crop(
                    x, segs[0], min_crop_size
                )
                cropped_segs = [
                    s[crop_d[0]:crop_d[1], crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
                    for s in segs
                ]
                return x_crop, cropped_segs
            else:
                return x, segs

        # Single segmentation case
        else:
            if cropping:
                x_crop, y_crop, _, _, _ = self._random_3d_crop(
                    x, segs, min_crop_size
                )
                return x_crop, y_crop
            else:
                return x, segs


    
    def _process_dimension(self, x, y, d, offset, max_number_labels, x_new, y_new, prompt, offset_list, slices_added, max_data_points):
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

        x_2d, x_2d_r, total_label, total_label_r = self._cast_norm_resize(
            x_2d, x_2d_r, total_label, total_label_r
        )

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
    
    def _get_data_points(self, max_data_points, offset, max_number_labels, dimensions, cropping, min_crop_size):
        start = time.time()
        print("Creating new Data Points ...")

        offset_list = []
        x_new, y_new, prompt = [], [], []
        slices_added = 0

        while slices_added < max_data_points:

            random.shuffle(self.dataloader.current_ids)

            for id in self.dataloader.current_ids:

                current_dict = self.dataloader.dataset[id]
                x, y = self._prepare_volume(current_dict, cropping, min_crop_size)
                
                # Unwrap single-element list to treat as a single multi-label volume
                if isinstance(y, list) and len(y) == 1:
                    y = y[0]
                    
                random.shuffle(dimensions)

                for d in dimensions:

                    if slices_added >= max_data_points:
                        break

                    slices_added = self._process_dimension(
                        x, y, d, offset, max_number_labels,
                        x_new, y_new, prompt, offset_list, slices_added, 
                        max_data_points)

                    if slices_added >= max_data_points:
                        break

        # Finalize dataset
        x_new, y_new, prompt, offset_list = self._randomnizer(x_new, y_new, prompt, offset_list, dimensions)
        ds = tf.data.Dataset.from_tensor_slices((
            tf.stack(x_new), tf.stack(y_new), tf.stack(prompt)
        ))

        print(f'It took {time.time() - start:.0f} seconds')
        return ds, offset_list

    
    def _get_data_points_from_one_task(self, max_data_points, offset, dimensions, cropping, min_crop_size):
        
        start = time.time()
        print("Creating new Data Points ...")

        offset_list = []
        x_new, y_new, prompt = [], [], []
        slices_added = 0
        step_counter = 0
        task = 0

        random.shuffle(self.dataloader.current_ids)
        random.shuffle(dimensions)
        d = dimensions[0]

        while slices_added < max_data_points:

            if step_counter >= 400:
                # Reset everything if took too long
                task = 0
                step_counter = 0
                slices_added = 0
                offset_list.clear()
                x_new.clear()
                y_new.clear()
                prompt.clear()
                print("Changed the task, because it took too long.")

            for id in self.dataloader.current_ids:
                if slices_added >= max_data_points:
                    break

                current_dict = self.dataloader.dataset[id]
                x, y = self._prepare_volume(current_dict, cropping, min_crop_size)

                # Unwrap single-element list to treat as a single multi-label volume
                if isinstance(y, list) and len(y) == 1:
                    y = y[0]

                if isinstance(y, list):
                    y_shape = y[0].shape['xyz'.index(d)]
                else:
                    y_shape = y.shape['xyz'.index(d)]
                iter_2d = tf.range(y_shape)
                tf.random.shuffle(iter_2d)

                for i in iter_2d:
                    step_counter += 1
                    if slices_added >= max_data_points:
                        break

                    r = self._sample_offset(i, offset, y_shape)
                    if r is None:
                        continue

                    if task == 0:
                        if isinstance(y, list):
                            task = random.randint(1, len(y))
                        else:
                            y_flat = tf.reshape(y, [-1])
                            valid_tasks, _ = tf.unique(y_flat)
                            valid_tasks = valid_tasks.numpy().tolist()
                            if 0 in valid_tasks:
                                valid_tasks.remove(0)
                            if len(valid_tasks) == 0:
                                continue
                            task = random.choice(valid_tasks)
                        print(f'Current task: {task}')

                    if isinstance(y, list):
                        y_2d = self._get_2d_data(y[task-1], i, d)
                        y_2d_r = self._get_2d_data(y[task-1], i + r, d)
                    else:
                        y_2d = self._get_2d_data(y, i, d)
                        y_2d_r = self._get_2d_data(y, i + r, d)

                    # Task label verification check
                    y1 = tf.reshape(y_2d, (-1,))
                    y2 = tf.reshape(y_2d_r, (-1,))
                    valid_labels, _ = tf.unique(y1)
                    valid_labels_r, _ = tf.unique(y2)

                    if isinstance(y, list):
                         # For binary masks in a list, we usually look for value 1
                         label = tf.where(y_2d > 0, 1, 0)
                         label_r = tf.where(y_2d_r > 0, 1, 0)
                    else:
                        if task not in valid_labels or task not in valid_labels_r:
                            continue
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

                    x_2d, x_2d_r, total_label, total_label_r = self._cast_norm_resize(
                        x_2d, x_2d_r, total_label, total_label_r
                    )
                    p = tf.concat([x_2d_r, total_label_r], axis=-1)

                    x_new.append(x_2d)
                    y_new.append(total_label)
                    prompt.append(p)
                    offset_list.append(r)
                    slices_added += 1

        # Shuffle / stack
        x_new, y_new, prompt, offset_list = self._randomnizer(
            x_new, y_new, prompt, offset_list, dimensions
        )
        ds = tf.data.Dataset.from_tensor_slices((
            tf.stack(x_new), tf.stack(y_new), tf.stack(prompt)
        ))

        # Clean up
        x_new, y_new, prompt = [], [], []

        print(f'It took {(time.time() - start):.0f} seconds')
        return ds, offset_list
    
    
    # -------------------------- Public Getters --------------------------
    
    def get_data_points(self, max_data_points=3500, offset=5, max_number_labels=1, dimensions=['x','y','z'], cropping=False, min_crop_size=0.5, cropping_composition=1):
        "If cropping_composition is e.g. 0.8, then 80% cropped dp are created and 20% non cropped"
        
        self.dataloader.current_ids = self.dataloader.train_ids
        
        if cropping_composition == 1:
            
            ds, offsets = self._get_data_points(max_data_points, offset, max_number_labels, dimensions, cropping, min_crop_size)
            return ds, offsets
        
        else:
            if cropping == False:
                raise ValueError("Can't use mixed cropping and no cropping at the same time. Set cropping_composition=1 or set cropping=True!")
                
            ds_crop, offset_crop = self._get_data_points(int(max_data_points*cropping_composition), offset, max_number_labels, dimensions, cropping, min_crop_size)
            ds_no_crop, offset_no_crop = self._get_data_points(int(max_data_points*(1.0-cropping_composition)), offset, max_number_labels, dimensions, cropping, min_crop_size)

            offset_crop = tf.data.Dataset.from_tensor_slices(offset_crop)
            offset_no_crop = tf.data.Dataset.from_tensor_slices(offset_no_crop)

            ds_crop = tf.data.Dataset.zip((ds_crop, offset_crop))
            ds_no_crop = tf.data.Dataset.zip((ds_no_crop, offset_no_crop))

            combined_ds = ds_crop.concatenate(ds_no_crop)

            combined_ds = combined_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=False)

            final_ds = combined_ds.map(lambda data, offset: data)
            final_offset_list = [offset.numpy() for _, offset in combined_ds]

            
            return final_ds, final_offset_list

    def get_val_data_points(self, max_data_points=3500, offset=5, max_number_labels=1, dimensions=['x','y','z'], cropping=False, min_crop_size=0.75):
        self.dataloader.current_ids = self.dataloader.validation_ids
        ds, offsets = self._get_data_points(max_data_points, offset, max_number_labels, dimensions, cropping, min_crop_size)
        return ds, offsets
    
    def get_data_points_from_one_task(self, max_data_points=3500, offset=5, dimensions=['x','y','z'], cropping=False, min_crop_size=0.75):
        """Be sure to choose the dimension, otherwise it will be all 3 mixed. And define the task if needed, otherwise it will be random.
        Returns: tf.Dataset. One Task from one random patient. Each element of ds = (x,y,p)
        """
        self.dataloader.current_ids = self.dataloader.train_ids
        ds, offsets = self._get_data_points_from_one_task(max_data_points, offset, dimensions, cropping, min_crop_size)
        return ds, offsets