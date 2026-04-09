from data.DataLoader import DataLoader
import os
import numpy as np

from utils.ds_handler import load_dataset

class DataLoader_npz(DataLoader):
    """
    npz_paths: Paths to .npz files. Create them with utils.ds_handler.save_dataset()
    val_size: Percentage of Patients to be used for validation. Use 0.0 when no validation run for training is made.
    mode: No effect
    max_img: No effect
    """

    def __init__(self, npz_paths, val_size, mode="", max_img=10000):
        # Resolve paths relative to the location of the script (project root)
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent  # folder containing DataLoader_npz.py
        project_root = script_dir.parent

        self.npz_paths = [str((project_root / p).resolve()) for p in npz_paths]

        super().__init__(val_size=val_size, mode=mode, max_img=max_img)

    # --------------------------------------------------------------

    def _to_numpy(self, x):
        """Convert tensor, list, nib or array into numpy array."""
        import tensorflow as tf
        import nibabel as nib

        if isinstance(x, np.ndarray):
            return x
        if tf.is_tensor(x):
            return x.numpy()
        if hasattr(x, "get_fdata"):  # nibabel
            return x.get_fdata()
        if isinstance(x, list):
            return np.array(x)
        raise TypeError(f"Unsupported data type: {type(x)}")

    # --------------------------------------------------------------

    def _get_segmentation_list(self, segs):
        """
        Ensures a list of 3D numpy arrays is returned.
        Handles cases where data is stored as a single 4D array (Channels, H, W, D).
        """
        # 1. Standard List/Tuple Case
        if isinstance(segs, (list, tuple)):
            return [self._to_numpy(s) for s in segs]
        
        # 2. Dictionary Case
        if isinstance(segs, dict):
            return [self._to_numpy(s) for s in segs.values()]
        
        # 3. Single Array Case
        arr = self._to_numpy(segs)
        
        # Check if it is a 4D array (Channel-first assumption)
        if arr.ndim == 4:
            return [arr[i] for i in range(arr.shape[0])]
            
        return [arr]

    # --------------------------------------------------------------

    def _pull_data(self):
        """
        Loads all .npz files, namespaces PIDs, 
        and fills self.dataset in DataLoader format.
        """

        print("\nLoading NPZ dataset(s)…")

        for npz_path in self.npz_paths:
            if not os.path.exists(npz_path):
                print(f"WARNING: File does not exist: {npz_path}")
                continue

            # Load npz via ds_handler
            try:
                npz_data = load_dataset(npz_path)
            except Exception as e:
                print(f"ERROR reading {npz_path}: {e}")
                continue

            print(f"Loaded {len(npz_data)} PIDs from {npz_path}")

            prefix = os.path.splitext(os.path.basename(npz_path))[0]  # file name
            count = 0

            # Convert each item
            for pid, item in npz_data.items():
                if count >= self.max_img:
                    break
                count += 1

                pid = f"{prefix}_{pid}"   # namespace the pid

                if "image" not in item:
                    print(f"WARNING: PID {pid} has no 'image'")
                    continue

                img = self._to_numpy(item["image"])

                # Find segmentation(s)
                if "segmentations" in item:
                    seg_list = self._get_segmentation_list(item["segmentations"])
                elif "segmentation" in item:
                    seg_list = self._get_segmentation_list(item["segmentation"])
                else:
                    seg_list = []
                    print(f"WARNING: PID {pid} has no segmentation")

                # Store into dataset buffer
                self.dataset[pid] = {
                    "image": img,
                    "segmentations": seg_list
                }

        print(f"\nFinal dataset size: {len(self.dataset)} patients.\n")

    # --------------------------------------------------------------

    def _data_to_dict(self, tset, lset, iset):
        pass 
